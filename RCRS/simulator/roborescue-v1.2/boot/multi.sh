#! /bin/bash

for ((i=1;i<10;i++)); do
	./start.sh -m ../../test/map_$i -c ./config
done
