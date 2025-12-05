sinfo_t_idle

salloc -p dev_cpuonly -N 1 -t 1:00:00 --mem=230gb

salloc -p cpuonly -N 1 -t 12:00:00 --mem=300gb

salloc -p large -N 1 -n 8 -t 12:00:00 --mem=3000gb



salloc -p dev_accelerated -t 0:59:00 --gres=gpu:1

salloc -p accelerated -t 8:00:00 --gres=gpu:4


salloc -p dev_accelerated-h100 -t 0:59:00 --gres=gpu:4



kit_project_usage

/usr/lpp/mmfs/bin/mmlsquota -j $PROJECT_GROUP --block-size G -C hkn.scc.kit.edu hkfs-home
/usr/lpp/mmfs/bin/mmlsquota -u $(whoami) --block-size G -C hkn.scc.kit.edu hkfs-work


ws_allocate myspace 30
/hkfs/work/workspace/scratch/hgf_mxv5488-myspace

ws_list
ws_find myspace

# For folders created since 2022
for dir in src/transformers/models/*/; do
    git log --diff-filter=A --follow --format="%aD | $dir" -1 -- "$dir" | grep "2024"
done


exit
exit
exit
exit



model_name = "steph0713/deepffnllama-768_6_4-2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)


watch -n 10 -d squeue