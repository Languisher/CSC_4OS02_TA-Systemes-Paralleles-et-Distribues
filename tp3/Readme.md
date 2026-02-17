# TP3 - Bucket Sort MPI (Python)

## Exécution

Depuis la racine du projet :

```bash
mpiexec -n 4 uv run python tp3/bucket_sort_mpi.py -n 100 --print-array
```

Des options ....

- `-n, --size` : nombre total d'éléments à trier.
- `--min-value` : borne basse des valeurs générées.
- `--max-value` : borne haute des valeurs générées.
- `--seed` : graine aléatoire.
- `--print-array` : affiche les tableaux avant/après tri sur `rank 0`.
