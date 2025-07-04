import click
import os
import json
from jinja2 import Environment, FileSystemLoader
from src.hpo.optimize import run_hpo

@click.group()
def cli():
    """End-to-End Local MLOps Pipeline for Project Mini-T."""
    pass

@cli.command()
def setup():
    """프로젝트 초기 설정을 수행합니다."""
    click.echo("🚀 프로젝트 초기 설정을 시작합니다...")
    os.makedirs("configs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    click.echo("✅ 기본 폴더 구조를 확인 및 생성했습니다.")

@cli.command()
def hpo():
    """(Phase 4) 하이퍼파라미터 최적화를 실행합니다."""
    click.echo("🏃‍♂️ HPO를 시작합니다...")
    try:
        run_hpo()
        click.secho("✅ HPO 성공적으로 완료!", fg="green")
    except Exception as e:
        click.secho(f"🔥 HPO 실행 중 오류 발생: {e}", fg="red")

@cli.command(name="generate-report")
def generate_report():
    """(Phase 5) 실험 결과를 바탕으로 리포트를 생성합니다."""
    click.echo("🏃‍♂️ 최종 리포트를 생성합니다...")

    reports_dir = "reports"
    template_name = "report_template.md"
    template_path = os.path.join(reports_dir, template_name)

    os.makedirs(reports_dir, exist_ok=True)

    # 템플릿 파일이 없으면 기본 템플릿 생성
    if not os.path.exists(template_path):
        default_template = """# {{ project_name }} 최종 실험 결과 보고서

## 하이퍼파라미터 최적화 (HPO) 결과

* **최종 손실 (Best Loss):** `{{ best_loss }}`

### 최적 하이퍼파라미터 조합
```json
{{ hyperparameters_json_string }}
```"""
        with open(template_path, "w") as f:
            f.write(default_template)
        click.echo(f"기본 리포트 템플릿 '{template_name}'을 생성했습니다.")

    env = Environment(loader=FileSystemLoader(reports_dir))
    template = env.get_template(template_name)

    results_path = "hpo_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            best_params = json.load(f)
    else:
        click.secho(
            f"경고: '{results_path}' 파일을 찾을 수 없어 더미 데이터로 리포트를 생성합니다.",
            fg="yellow",
            err=True
        )
        best_params = {"loss": "N/A", "config": {"error": "hpo_results.json not found."}}

    # 딕셔너리를 보기 좋은 JSON 문자열로 변환
    hyperparameters_str = json.dumps(best_params.get("config", {}), indent=2)

    rendered = template.render(
        project_name="Project Mini-T",
        best_loss=best_params.get("loss", "N/A"),
        hyperparameters_json_string=hyperparameters_str
    )

    output_path = "final_report.md"
    with open(output_path, "w") as f:
        f.write(rendered)

    click.echo(f"✅ 최종 리포트('{output_path}')가 생성되었습니다.")

@cli.command()
@click.pass_context
def all(ctx):
    """모든 파이프라인 단계를 순서대로 실행합니다."""
    click.echo("🚀 전체 파이프라인을 시작합니다.")
    ctx.invoke(setup)
    ctx.invoke(hpo)
    ctx.invoke(generate_report)
    click.echo("🎉 모든 파이프라인 실행이 완료되었습니다!")

if __name__ == "__main__":
    cli()
