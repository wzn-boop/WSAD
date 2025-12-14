for num_epochs in 10 20 30
do
  for batch_size in 32 64 128
  do
    for lr in 0.00003 0.00005 0.00008
    do
      for seq_len in 30 50 80 100
      do
        for hidden_dims in '100' '100,50' '100,50,50'
        do
          python -u main.py --network TCNAE --rep False --objective MSE --num_epochs $num_epochs --batch_size $batch_size --lr $lr --seq_len $seq_len --hidden_dims $hidden_dims --act ReLU &
        done
        wait
      done
    done
  done
done