#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 file1 [file2 ... fileN]"
  exit 1
fi

total_size=0

for file in "$@"
do
  if [ -e "$file" ]; then
    size=$(du -k "$file" | awk '{print $1}')
    total_size=$(echo "$total_size + $size" | bc)
  else
    echo "File $file does not exist."
  fi
done

echo "Total size: ${total_size}K"
