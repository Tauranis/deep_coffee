install:	
	chmod +x scripts/download_dataset.sh
	bash ./scripts/download_dataset.sh
	
	chmod +x scripts/build_docker_image.sh
	bash ./scripts/build_docker_image.sh

	chmod +x scripts/preprocess_dataset.sh
	bash ./scripts/preprocess_dataset.sh