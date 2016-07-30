# Audio-driven, similarity-based Speech/Music discrimination of multimedia content using supervised learning

### Supported OS

* OS X
* Linux

### Dependencies

* Install Docker: https://docs.docker.com/installation/
* Verify that docker is working properly by running the hello-world docker container: ```docker run hello-world```

**Note**

The ```-v host-machine-directory-or-file:container-directory-or-file``` docker parameter used in the following commands allows to mount a directory from the host machine to the docker container in order to be able to process its contents and return the generated results back to the host machine.  

### Usage 

* Pull the algorithm's docker image from docker-hub:
```
docker pull nicktgr15/similarity-based-speech-music-discrimination
```

* Provide a multimedia file as input and analyse it using the following command syntax:

```docker run --rm -v /abs/path/to/input/dir:/abs/path/to/input/dir nicktgr15/similarity-based-speech-music-discrimination python /opt/speech-music-discrimination/speech-music-discriminator.py --input-file /abs/path/to/input/file```

* Results will become available under ```/abs/path/to/input/dir/annotated-segments.txt```.

##### Example:

If the multimedia input file is located under **/var/my/data/input.mp4** then the docker command would be:

```docker run --rm -v /var/my/data/:/var/my/data/ nicktgr15/speech-music-discriminatior python /opt/speech-music-discrimination/speech-music-discriminator.py --input-file /var/my/data/mp4```

### Troubleshooting

#### Cannot attach directories from the host machine
In OS X there is an issue when trying to attach directories outside the /Users dir. For example attaching host's /tmp directory to the container is not possible.

#### Running out of memory or out of disk space (OS X only)
If the virtualbox VM in which the containers are executed under OS X is running out of resources then you need to remove the existing VM and create a new one with more resources. 

*e.g.*
```
docker-machine rm default
docker-machine create --virtualbox-disk-size 40000 -virtualbox-memory 2048 -d virtualbox default
docker-machine ls
docker-machine start default
```
You may have to set the environment variables for the new VM by running ```eval "$(docker-machine env default)"```
