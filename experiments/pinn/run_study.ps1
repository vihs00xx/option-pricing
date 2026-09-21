$ErrorActionPreference = "Continue"
$log = Join-Path $PSScriptRoot "..\..\study_run.log"
function Log($m) { "{0} {1}" -f (Get-Date -Format o), $m | Out-File $log -Append }

Log "START study"

$seeds = @(19102026, 19112026, 19122026)
$tags = @("s0", "s1", "s2")
$variants = @("baseline_matched", "constrained", "causal")

for ($i = 0; $i -lt $seeds.Count; $i++) {
    foreach ($v in $variants) {
        Log "TRAIN $v seed=$($seeds[$i]) tag=$($tags[$i])"
        python -m experiments.pinn.train_pinn_variant --variant $v --seed $seeds[$i] --tag $tags[$i] *>> $log
        Log "TRAIN_DONE $v tag=$($tags[$i]) exit=$LASTEXITCODE"
    }
}

Log "REFINE lbfgs from baseline s0 adam"
python -m experiments.pinn.refine_pinn_optimizer --optimizer lbfgs --start checkpoints/pinn/research_variants/matched_study/study_baseline_matched_s0_adam.pt *>> $log
Log "REFINE_DONE lbfgs exit=$LASTEXITCODE"

Log "REFINE nys_newton_cg from baseline s0 adam"
python -m experiments.pinn.refine_pinn_optimizer --optimizer nys_newton_cg --start checkpoints/pinn/research_variants/matched_study/study_baseline_matched_s0_adam.pt *>> $log
Log "REFINE_DONE nys_newton_cg exit=$LASTEXITCODE"

Log "EVALUATE"
python -m experiments.pinn.evaluate_pinn_study_multi *>> $log
Log "EVALUATE_DONE exit=$LASTEXITCODE"

Log "DONE study"
