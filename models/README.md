
How to run interactive debugging job:
1. `qsub -I -P MLFluids -l select=1:ncpus=1:mem=4gb -l walltime=1:00:00`
2. `module load python/3.8.2 cuda/10.2.89 magma/2.5.3`
3. `source /project/MLFluids/pytorch_1.11/bin/activate`
4. `CUDA_VISIBLE_DEVICES=0 python3 /home/cfos3120/PINN/gnot_trainer.py`