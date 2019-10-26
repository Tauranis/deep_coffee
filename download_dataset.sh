RAW_FILENAME="dataset/raw.zip"
RAW_DATASET_FILE_ID="1qg7n_uLsMD5wa5LrVVHF6Gp0W8U5RcKN"

CROPPED_FILENAME="dataset/cropped.zip"
CROPPED_DATASET_FILE_ID="1Ctb3Mx3VM11PUHvkcOGWvwhqq1ThQ8Bq"

download_and_extract(){
    # Download dataset
    filename=$1
    fileid=$2    

    echo "Download dataset ${filename}"
    query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${fileid}" \
    | perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
    | sed -e 's/amp;//g' | sed -n 2p`
    url="https://drive.google.com$query"
    curl -b ./cookie.txt -L -o ${filename} ${url}

    echo "Cleaning up"
    rm ./cookie.txt    
    unzip ${filename} -d ./dataset
}


# Create dataset directory
echo "Create dataset directory structure"
mkdir -p dataset
# mkdir -p dataset/train/good
# mkdir -p dataset/train/bad
# mkdir -p dataset/eval/good
# mkdir -p dataset/eval/bad
# mkdir -p dataset/test/good
# mkdir -p dataset/test/bad

# # Download dataset
download_and_extract ${RAW_FILENAME} ${RAW_DATASET_FILE_ID}
download_and_extract ${CROPPED_FILENAME} ${CROPPED_DATASET_FILE_ID}

echo "Done"

