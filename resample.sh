#!/bin/bash

# give the directory path as first argument and new sample rate as second one and this script will resample all wav files

cd $1
echo "your now in "`pwd`
mkdir -p $2
for i in *.wav;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  #ffmpeg -i "$i" "${name}.wav"
  sox "$i" -r $2 $1/$2/"$i"
  #rm $i
done