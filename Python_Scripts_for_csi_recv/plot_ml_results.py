import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config

from csi_parser import configure_console_output
configure_console_output()


def main():
    defaults = config.get_script_defaults("plot_ml_results")
    parser = argparse.ArgumentParser(description="Plot ML metrics from metrics.json for Thesis")
    parser.add_argument("--json_path", type=str, default=defaults["json_path"])
    parser.add_argument("--out_dir", type=str, default=defaults["out_dir"])
    config.add_bool_argument(
        parser,
        dest="save",
        default=defaults["save"],
        help="Save plots as PNG",
        positive_flags=["--save"],
        negative_flags=["--no-save"],
    )
    config.add_bool_argument(
        parser,
        dest="show",
        default=defaults["show"],
        help="Show plots interactively",
        positive_flags=["--show"],
        negative_flags=["--no-show"],
    )
    args = parser.parse_args()


    json_path = Path(args.json_path)
    out_dir = Path(args.out_dir)


    if not json_path.exists():
        print(f"[ERROR] Mising file: {json_path}")
        print("   Make sure you have run the pipeline with the --save_model flag:")
        print("   python csi_ml_pipeline.py --classes walk idle --save_model")
        return


    with open(json_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)


    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
    

    # Define aesthetic settings for thesis (clean and professional)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    

    print(f"\n[INFO] Generating publication-ready plots for {len(metrics)} models...")


    models = []
    test_accs = []
    f1_scores = []


    for model_name, data in metrics.items():
        cm = np.array(data['confusion_matrix'])
        classes = data['classes']
        test_acc = data['test_accuracy'] * 100
        test_f1 = data['test_f1_macro'] * 100


        models.append(model_name)
        test_accs.append(test_acc)
        f1_scores.append(test_f1)


        # ----------------------------------------------------
        # 1. Plot Confusion Matrix
        # ----------------------------------------------------
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                         cbar=False, square=True, 
                         xticklabels=classes, yticklabels=classes,
                         annot_kws={"size": 16, "weight": "bold"})
        

        plt.title(f"Confusion Matrix: {model_name}\nAcc: {test_acc:.1f}% | F1: {test_f1:.1f}%", 
                  pad=15, fontweight='bold', fontsize=14)
        plt.ylabel('True Class', fontweight='bold', fontsize=12)
        plt.xlabel('Predicted Class', fontweight='bold', fontsize=12)
        

        # Ensure labels are rotated nicely
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        

        if args.save:
            safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            img_path = out_dir / f"CM_{safe_name}.png"
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved {img_path.name}")
        
        if not args.show:
            plt.close()
        

        # ----------------------------------------------------
        # 1.5 Plot Feature Importances (if available)
        # ----------------------------------------------------
        importances = data.get('feature_importances', [])
        if importances:
            plt.figure(figsize=(8, 6))
            

            # Sort importances (already sorted from pipeline, but ensure ascending for barh)
            names = [item['name'] for item in importances][::-1]
            vals = [item['importance'] * 100 for item in importances][::-1]
            

            ax = sns.barplot(x=vals, y=names, hue=names, palette="viridis", legend=False)
            plt.title(f"Top 10 Features: {model_name}", pad=15, fontweight='bold', fontsize=14)
            plt.xlabel('Importance (%)', fontweight='bold', fontsize=12)
            plt.ylabel('Feature', fontweight='bold', fontsize=12)
            

            # Add value labels to bars
            for p, val in zip(ax.patches, vals):
                if p.get_width() > 0:
                    ax.annotate(f"{val:.1f}%",
                                (p.get_width() + 0.5, p.get_y() + p.get_height() / 2.),
                                ha='left', va='center', fontweight='bold',
                                color='#333333', fontsize=10)
            

            sns.despine()
            plt.tight_layout()
            

            if args.save:
                feat_img_path = out_dir / f"Features_{safe_name}.png"
                plt.savefig(feat_img_path, dpi=300, bbox_inches='tight')
                print(f"  [OK] Saved {feat_img_path.name}")
            
            if not args.show:
                plt.close()


    # ----------------------------------------------------
    # 2. Plot Comparison Bar Chart (if multiple models)
    # ----------------------------------------------------
    if len(models) > 1:
        x = np.arange(len(models))
        width = 0.35


        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', color='#4C72B0')
        rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Macro Score', color='#DD8452')


        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight='bold', fontsize=12)
        

        # Style the legend
        ax.legend(loc='lower right', frameon=True, shadow=True)
        ax.set_ylim([0, 110]) # Leave room for labels on top


        # Add text labels on top of bars
        ax.bar_label(rects1, fmt='%.1f%%', padding=3, fontweight='bold', color='#333333')
        ax.bar_label(rects2, fmt='%.1f%%', padding=3, fontweight='bold', color='#333333')


        # Remove top/right borders for a cleaner look
        sns.despine()


        fig.tight_layout()
        if args.save:
            comp_path = out_dir / "Model_Comparison.png"
            plt.savefig(comp_path, dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved {comp_path.name}")
        
        if not args.show:
            plt.close()


    if args.save:
        print(f"\n[OK]  Plots saved in: {out_dir.absolute()}\n")

    if args.show:
        print("[INFO] Showing plots. Close all windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
