#!/bin/bash

# give the directory path as first argument and this script will convert all mp3 files in it to wav

cd $1
echo "your now in "`pwd`
for i in *.mp3;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  ffmpeg -i "$i" "${name}.wav"
  rm $i
done