import click
import subprocess
import sys

def run_command(command_args):
    """주어진 명령어를 실행하고 성공 여부를 반환하는 헬퍼 함수."""
    try:
        module_path = command_args[1].replace('/', '.').replace('\\', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        
        command = [sys.executable, "-m", module_path]
        
        click.echo(f"--- Running command: {' '.join(command)} ---")
        
        # 실시간으로 출력을 스트리밍하기 위해 Popen 사용
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            if p.stdout:
                for line in p.stdout:
                    click.echo(line, nl=False)
        
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, p.args)

        click.echo(f"--- Command successful ---")
        return True
    except subprocess.CalledProcessError:
        click.secho("--- Command FAILED ---", fg='red', bold=True)
        return False
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red', bold=True)
        return False

# ### 수정된 부분: @click.pass_context 추가 ###
@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
def cli(ctx):
    """
    Project Mini-T-TYPE-2: End-to-End MLOps Pipeline
    """
    # 컨텍스트 객체에
    # 공통 변수 등을 저장할 수 있습니다.
    ctx.obj = {}

@cli.command()
def check_sanity():
    """Phase 3: GeometricPIDNet의 건전성을 검사합니다."""
    if not run_command(["python", "scripts/check_geometric_pid.py"]):
        sys.exit(1)

@cli.command()
def train_geometric_pid():
    """Phase 3: 최종 GeometricPIDNet 모델을 훈련하고 저장합니다."""
    if not run_command(["python", "scripts/run_phase3_training.py"]):
        sys.exit(1)

@cli.command()
def run_hpo():
    """Phase 2: Colab에서 HPO를 실행하라는 안내를 출력합니다."""
    click.echo(click.style("INFO: HPO is designed to run on Google Colab with a GPU.", fg="yellow"))
    click.echo("Please run the following command in your Colab notebook:")
    click.echo(click.style("!python -m src.hpo.optimize", fg="green"))

@cli.command()
def train_final_model():
    """Phase 2의 HPO 결과로 최종 PIDNet을 훈련하고 저장합니다."""
    click.echo(click.style("INFO: This script uses the best hyperparameters found by HPO.", fg="cyan"))
    if not run_command(["python", "scripts/train_final_model.py"]):
        sys.exit(1)

# ### 수정된 부분: ctx.invoke를 사용하여 다른 명령어를 호출 ###
@cli.command()
@click.pass_context
def run_all(ctx):
    """전체 파이프라인을 순차적으로 실행합니다 (HPO 제외)."""
    click.echo(click.style("===== STARTING FULL PIPELINE =====", bold=True, fg='blue'))
    
    try:
        click.echo("\n[Step 1/3] Running GeometricPIDNet Sanity Check...")
        ctx.invoke(check_sanity)
        
        click.echo("\n[Step 2/3] Training the final standard PIDNet model...")
        ctx.invoke(train_final_model)
            
        click.echo("\n[Step 3/3] Training the final GeometricPIDNet model...")
        ctx.invoke(train_geometric_pid)

        click.echo(click.style("\n===== FULL PIPELINE COMPLETED SUCCESSFULLY! =====", bold=True, fg='green'))

    except SystemExit as e:
        if e.code != 0:
             click.secho(f"\nPipeline stopped due to an error in the previous step.", fg='red', bold=True)

# cli.add_command 방식은 더 이상 필요 없습니다. @cli.command 데코레이터가 자동으로 추가해줍니다.

if __name__ == '__main__':
    cli()