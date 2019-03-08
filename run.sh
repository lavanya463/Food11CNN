sh clean.sh
sbatch launcher.cmd
for i in {1..7}
do
	squeue
	sleep 10
done
