julia -e 'using LiveServer; servedocs(foldername=pwd())' --project="." &
julia -e 'using DocumenterVitepress: dev_docs; dev_docs("build", md_output_path="")' --project="." &

wait