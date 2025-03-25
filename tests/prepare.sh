if [ ! -d "./fridge_env_pickles/" ]; then
    gdown --id 17WwQHI4sx7DeVet19oI2uUxjb-TnjXLZ -O data/file.tar.gz && tar -xzvf data/file.tar.gz -C data
fi

if [ ! -d "./mugcup_pose_pickles/" ]; then
    gdown --id 12TNBeidS7aUEjEN5Oa34dipSntEwS7ZP -O data/file.tar.gz && tar -xzvf data/file.tar.gz -C data
fi
