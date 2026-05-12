with open("scripts/build_refinement_litepp.py", "r") as f:
    code = f.read()

old_block = """            # Create PR pair if mismatch
            if student_action != pseudo_golden and student_action != "UNKNOWN":
                pr_item = {
                    "prompt": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_msg}
                    ],
                    "chosen": s["output"],  # Current teacher response containing pseudo_golden
                    "rejected": student_resp
                }
                out.write(json.dumps(pr_item) + "\\n")
                pr_items.append(pr_item)"""

new_block = """            # Create PR sample if mismatch (SFT Format)
            if student_action != pseudo_golden and student_action != "UNKNOWN":
                pr_item = {
                    "instruction": instruction,
                    "input": user_msg,
                    "output": s["output"]  # Current teacher response containing pseudo_golden (best_action_by_env)
                }
                out.write(json.dumps(pr_item) + "\\n")
                pr_items.append(pr_item)"""

code = code.replace(old_block, new_block)

with open("scripts/build_refinement_litepp.py", "w") as f:
    f.write(code)
