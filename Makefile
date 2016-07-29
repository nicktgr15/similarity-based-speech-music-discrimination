NAME=speech-music-discriminator

build:
	docker build -t $(NAME) .

run:
	docker run -it --rm=true $(NAME) bash

remove:
	-docker stop $(NAME)
	-docker rm $(NAME)

env:
	virtualenv env
	. env/bin/activate && pip install -r requirements.txt

extract_features_gtzan:
	docker run -w /workspace -v $(shell pwd):/workspace -it --rm=false nicktgr15/yaafe-docker yaafe -c datasets/featureplans/featureplan -r 22050 datasets/gtzan/*.wav

extract_features_labrosa:
	docker run -w /workspace -v $(shell pwd):/workspace -it --rm=false nicktgr15/yaafe-docker yaafe -c datasets/featureplans/featureplan -r 22050 datasets/labrosa/*.wav

extract_features_mirex:
	docker run -w /workspace -v $(shell pwd):/workspace -it --rm=false nicktgr15/yaafe-docker yaafe -c datasets/featureplans/featureplan -r 22050 datasets/mirex/*.wav

extract_input_features:
	docker run -w /workspace -v $(shell pwd):/workspace -it --rm=false nicktgr15/yaafe-docker yaafe -c datasets/featureplans/featureplan -r 22050 tmp/input.wav

extract_all_features: extract_features_gtzan extract_features_labrosa extract_features_mirex