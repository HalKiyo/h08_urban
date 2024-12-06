# map/dat
source_dir="/mnt/c/Users/tsimk/Downloads/H08_20230612/map/dat"
target_dir="/mnt/c/Users/tsimk/Downloads/dotfiles/h08/region/dat"

# ソースディレクトリ内のディレクトリ名を取得
directories=$(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")

# ターゲットディレクトリ内に存在しないディレクトリを作成
for dir_name in $directories; do
    if [ ! -d "$target_dir/$dir_name"  ]; then
        mkdir "$target_dir/$dir_name"
        echo "Created directory: $target_dir/$dir_name"
    else
        echo "Directory already exists: $target_dir/$dir_name"
    fi
done

# city data
source_dir="/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat"
target_dir="/mnt/c/Users/tsimk/Downloads/dotfiles/h08/region/dat"

# ソースディレクトリ内のディレクトリ名を取得
directories=$(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")

# ターゲットディレクトリ内に存在しないディレクトリを作成
for dir_name in $directories; do
    if [ ! -d "$target_dir/$dir_name"  ]; then
        mkdir "$target_dir/$dir_name"
        echo "Created directory: $target_dir/$dir_name"
    else
        echo "Directory already exists: $target_dir/$dir_name"
    fi
done

# met data
source_dir="/mnt/c/Users/tsimk/Downloads/H08_20230612/met/dat"
target_dir="/mnt/c/Users/tsimk/Downloads/dotfiles/h08/region/dat"

# ソースディレクトリ内のディレクトリ名を取得
directories=$(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n")

# ターゲットディレクトリ内に存在しないディレクトリを作成
for dir_name in $directories; do
    if [ ! -d "$target_dir/$dir_name"  ]; then
        mkdir "$target_dir/$dir_name"
        echo "Created directory: $target_dir/$dir_name"
    else
        echo "Directory already exists: $target_dir/$dir_name"
    fi
done
