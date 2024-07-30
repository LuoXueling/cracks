while getopts "p:n:r:" opt
do
   case "$opt" in
      p ) path="$OPTARG" ;;
      n ) nprocesses="$OPTARG" ;;
      r ) release_or_debug="$OPTARG" ;;
   esac
done

path="${path:-build}"
nprocesses="${nprocesses:-8}"
release_or_debug="${release_or_debug:-release}"

rm -rf "$path"
mkdir "$path"
cd "$path"
cmake ..
make "$release_or_debug"
make
mpirun -n "$nprocesses" ./main
