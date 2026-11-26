# MaintiDoc — demonstration
# Usage :  powershell -ExecutionPolicy Bypass -File demo.ps1

$ErrorActionPreference = "Continue"
$env:HF_HOME       = "D:\hf_cache"
$env:UV_CACHE_DIR  = "D:\uv_cache"
$env:OLLAMA_MODELS = "D:\ollama\models"
# qwen2.5vl:3b tient dans les 4 Go de VRAM (29/37 couches sur GPU, ~26 t/s).
# Le 7b n'y place que 2 couches sur 29 et tombe a ~3 t/s.
$env:MAINTIDOC_VLM_MODEL = "qwen2.5vl:3b"

Set-Location -Path $PSScriptRoot

$QUESTION = "Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?"

function Show-Title($n, $text) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host ("  $n. $text") -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host ""
}

Show-Title 1 "Tests : 192 tests hors modeles, sans reseau"
uv run --offline pytest -q

Show-Title 2 "L'index est reel : 41 documents indexes"
uv run --offline python -m maintidoc.retrieval stats

Show-Title 3 "SANS reranker  ->  le mauvais modele arrive en tete"
Write-Host "   Question : $QUESTION" -ForegroundColor Yellow
Write-Host ""
uv run --offline python -m maintidoc.retrieval ask $QUESTION --no-rerank 2>$null

Show-Title 4 "AVEC reranker  ->  le bon modele (SAN-LAB 2000) passe en tete"
Write-Host "   Meme question, seul le reranker change." -ForegroundColor Yellow
Write-Host ""
uv run --offline python -m maintidoc.retrieval ask $QUESTION 2>$null

Show-Title 5 "Ollama et le modele vision-langage"
uv run --offline python -m maintidoc.generation check

Show-Title 6 "Generation sourcee : reponse + verification des citations"
Write-Host "   Environ 40 s : le modele tourne en partie sur CPU." -ForegroundColor Yellow
Write-Host ""
uv run --offline python -m maintidoc.generation answer $QUESTION 2>$null

Write-Host ""
Write-Host ("=" * 78) -ForegroundColor DarkCyan
Write-Host "  A RETENIR" -ForegroundColor Green
Write-Host ("=" * 78) -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  Dix modeles d'equipement emploient la meme formule pour le voyant"
Write-Host "  orange fixe, avec dix significations differentes. La fusion seule"
Write-Host "  place SAN-RACK 240 en tete ; seul le cross-encoder, qui lit la"
Write-Host "  question et le passage ensemble, remet SAN-LAB 2000 en premier."
Write-Host ""
Write-Host "  C'est pourquoi le reranker n'est pas optionnel, et c'est la ligne"
Write-Host "  de base (--no-rerank) sur laquelle la phase 10 mesurerait son"
Write-Host "  fine-tuning."
Write-Host ""

Read-Host "Appuyer sur Entree pour fermer"
