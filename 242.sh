for num_epochs in 10 20 30
do
  for batch_size in 32 64 128
  do
    for lr in 0.00003 0.00005 0.00008
    do
      for seq_len in 30 50 80 100
      do
        for rep_dim in 128 256 512
        do
          python -u main.py --network ConvSeqEn --rep_dim $rep_dim --num_epochs $num_epochs --batch_size $batch_size --lr $lr --seq_len $seq_len --hidden_dims 100 --act ReLU &
          python -u main.py --network CDTTransformerEn --attn cc_attn --rep_dim $rep_dim --num_epochs $num_epochs --batch_size $batch_size --lr $lr --seq_len $seq_len --hidden_dims 512 --act GELU &
        done
        wait
      done
    done
  done
done