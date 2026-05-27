with open("README_LITEPP.md", "r") as f:
    text = f.read()

text = text.replace("8. **End-to-End CityFlow Evaluation", "8. **Training PR Refinement through SFT**\n   Execute SFT directly using Kaggle or your local GPU across the PR configs constructed:\n   ```bash\n   llamafactory-cli train config/llamafactory_pr_qwen1_5b.yaml\n   ```\n\n9. **End-to-End CityFlow Evaluation")

with open("README_LITEPP.md", "w") as f:
    f.write(text)
