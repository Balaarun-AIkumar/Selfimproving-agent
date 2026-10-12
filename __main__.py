import argparse
import json
from pathlib import Path
import sys

from .agent import Agent
from .clinic import Clinic
from .evaluation import run_suite
from .learning import improve
from .policy import Policy


def port_number(value):
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Port must be an integer from 0 to 65535") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be from 0 to 65535")
    return port


def main():
    parser = argparse.ArgumentParser(description="CareLoop synthetic patient scheduling agent")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local chat and evaluation dashboard")
    serve.add_argument("--port", type=port_number, default=8000, metavar="PORT")
    serve.add_argument("--data-dir", help="Default: runtime/gemini for Gemini, runtime for offline")
    serve.add_argument("--gemini", action="store_true", help="Use the Gemini key in .env for language and improvements")
    chat = sub.add_parser("chat", help="Talk to the agent in your terminal")
    chat.add_argument("--policy")
    chat.add_argument("--baseline", action="store_true")
    chat.add_argument("--gemini", action="store_true")
    evaluate = sub.add_parser("eval", help="Run objective evaluation without changing policy")
    evaluate.add_argument("--split", choices=["training", "validation"], default="training")
    evaluate.add_argument("--policy")
    evaluate.add_argument("--output")
    evaluate.add_argument("--gemini", action="store_true")
    loop = sub.add_parser("improve", help="Run the full baseline → repair → replay → gate loop")
    loop.add_argument("--output", default="artifacts/latest")
    loop.add_argument("--policy", help="Optional initial policy; default is the frozen baseline")
    loop.add_argument("--gemini", action="store_true")
    loop.add_argument("--fresh", action="store_true", help="Explicitly start a Gemini experiment from the initial policy")
    loop.add_argument("--attempts", type=int, default=2)
    sub.add_parser('check-gemini', help='Make one small live request without printing the API key')
    args = parser.parse_args()
    client = interpreter = None
    if getattr(args, 'gemini', False) or args.command == 'check-gemini':
        from .gemini import GeminiClient
        from .language import GeminiInterpreter
        cache_path = None
        if args.command == 'improve':
            if args.output == 'artifacts/latest':
                args.output = 'artifacts/gemini/latest'
            cache_path = Path(args.output) / 'model-cache.json'
        client = GeminiClient(cache_path=cache_path, fresh=getattr(args, 'fresh', False))
        interpreter = GeminiInterpreter(client)
    if args.command == 'check-gemini':
        result = client.generate('Return {"connected":true}.', {},
                                 {'type':'OBJECT','properties':{'connected':{'type':'BOOLEAN'}},'required':['connected']})
        if result != {'connected': True}:
            raise ValueError('Gemini connection check returned an unexpected result')
        print(f'Gemini connected: {client.model}. Key kept server-side.')
        return 0
    if args.command in {"eval", "improve"} and args.policy and not Path(args.policy).is_file():
        parser.error(f"Policy file not found: {args.policy}")
    if args.command == "serve":
        from .server import serve
        serve(args.port, args.data_dir or ('runtime/gemini' if client else 'runtime'), client=client)
    elif args.command == "chat":
        clinic = Clinic()
        if not args.policy:
            args.policy = 'artifacts/gemini/latest/active_policy.json' if client else 'artifacts/latest/active_policy.json'
        agent = Agent(clinic, Policy() if args.baseline else Policy.load(args.policy), interpreter=interpreter)
        print(f"Policy {agent.policy.version}; clinic today {clinic.today}; type /quit to exit")
        print(agent.greeting)
        try:
            while True:
                try:
                    message = input("\nYou: ")
                except EOFError:
                    break
                if message.strip() == "/quit":
                    break
                try:
                    print("\nCareLoop: " + agent.reply(message))
                except ValueError as exc:
                    print(str(exc))
        finally:
            clinic.close()
    elif args.command == "eval":
        cases = None
        if client:
            from .gemini_learning import suite_cases
            cases = suite_cases(args.split)
        result = run_suite(Policy.load(args.policy) if args.policy else Policy(), args.split, cases, interpreter,
                           (lambda message: print(message, flush=True)) if client else None)
        print(f"{args.split}: {result['passed']}/{result['total']} ({result['score']}%); safety {'PASS' if result['safety_passed'] else 'FAIL'}")
        for case in result["results"]:
            print(f"{'PASS' if case['passed'] else 'FAIL'} {case['id']}" + (": " + ", ".join(case["failure_codes"]) if not case["passed"] else ""))
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if result["passed"] == result["total"] else 1
    elif args.command == "improve":
        if client:
            from .gemini_learning import improve_gemini
            if args.output == 'artifacts/latest':
                args.output = 'artifacts/gemini/latest'
            baseline = Policy.load(args.policy) if args.policy else (Policy() if args.fresh else Policy.load(Path(args.output) / 'active_policy.json'))
            result = improve_gemini(args.output, client, baseline, args.attempts,
                                    lambda message: print(message, flush=True),
                                    proposer_client=client.repair_client(), fresh=args.fresh)
        else:
            result = improve(args.output, Policy.load(args.policy) if args.policy else None)
        print(f"Training: {result['before']['score']}% -> {result['after']['score']}%")
        print(f"Validation: {result['validation_before']['score']}% -> {result['validation_after']['score']}%")
        for patch in result["proposal"]["patches"]:
            print(f"{patch['failure_code']}: {patch['field']} = {patch['value']}")
        print(f"Promotion: {'ACCEPTED' if result['gate']['promoted'] else 'REJECTED'}; regressions: {len(result['gate']['regressions'])}")
        print(f"Evidence: {Path(args.output).resolve()}")
        return 0 if result["gate"]["promoted"] else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Could not run CareLoop: {error}", file=sys.stderr)
        sys.exit(2)
