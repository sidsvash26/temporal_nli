## If copying script exists in the folder, run it
for d in */ ; do
    cd $d
       find . -type d -exec cp merges.txt {} \;
       find . -type d -exec cp vocab.json {} \; 
       find . -type d -exec cp special_tokens_map.json {} \;
       find . -type d -exec cp tokenizer_config.json {} \;
    cd ..
done
