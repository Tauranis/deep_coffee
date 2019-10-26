RAW_FILENAME="dataset/raw.zip"
RAW_DATASET_FILE_ID="1qg7n_uLsMD5wa5LrVVHF6Gp0W8U5RcKN"

# Create dataset directory
echo "Create dataset directory structure"
mkdir -p dataset
mkdir -p dataset/cropped
mkdir -p dataset/train/good
mkdir -p dataset/train/bad
mkdir -p dataset/eval/good
mkdir -p dataset/eval/bad
mkdir -p dataset/test/good
mkdir -p dataset/test/bad

# Download dataset
echo "Download dataset"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${RAW_DATASET_FILE_ID}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${RAW_FILENAME} ${url}

echo "Cleaning up"
rm ./cookie.txt
rm ./query.txt
unzip ./dataset/raw.zip -d ./dataset

echo "Done"