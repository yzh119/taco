for dataset in cora citeseer pubmed arxiv ppi
do
    echo "---------------dataset:" ${dataset} "------------------"
    for feat_size in 32 64 128 256 512
    do
        echo "-----feat size:" ${feat_size} "-----"
        ../build/bin/sddmm data/${dataset}.npz ${feat_size}
    done
done