while getopts "p:n:r:s:" opt
do
   case "$opt" in
      p ) path="$OPTARG" ;;
      n ) nprocesses="$OPTARG" ;;
      r ) release_or_debug="$OPTARG" ;;
      s ) skip_compile="$OPTARG" ;;
   esac
done

path="${path:-build}"
nprocesses="${nprocesses:-8}"
release_or_debug="${release_or_debug:-release}"
skip_compile="${skip_compile:-false}"

if [ "$skip_compile" = "true" ]; then
   cd "$path"
   cd ..
   cp "$path/main" ./main
   mpirun -n "$nprocesses" ./main
else
   rm -rf "$path"
   mkdir "$path"
   cd "$path"
   cmake ..
   make "$release_or_debug"
   make
   cd ..
   cp "$path/main" ./main
   mpirun --oversubscribe -n "$nprocesses" ./main 
fi
