#!/bin/bash
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH --job-name=PCA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=08:00:00
#SBATCH --partition=inferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

hostname

module load anaconda3
conda activate torsional_diffusion_2

python3 analyze_dist.py

end=`date +%s`
echo "Done   " `date`
runtime=$((end-start))
hours=$(printf %02d $((runtime / 3600)))
minutes=$(printf %02d $(( (runtime % 3600) / 60 )))
seconds=$(printf %02d $(( (runtime % 3600) % 60 )))
echo "Elapsed time: $hours:$minutes:$seconds (hh:mm:ss)"
echo "Elapsed time: $runtime seconds"
exit $ret


































































