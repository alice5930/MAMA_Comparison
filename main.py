"""
Main Entry Point
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from config import get_config
from src.orchestration import create_mama_workflow, process_flight_query_simple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mama_system.log"),
    ],
)

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAMA Flight Selection Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive flight query
  python main.py --interactive

  # Single flight query
  python main.py --departure "New York" --destination "Los Angeles" --date "2024-03-15"

  # Query with preferences
  python main.py --departure "Chicago" --destination "Miami" --date "2024-04-01" --priority "safety"

  # Full evaluation
  python main.py --evaluate

  # Comprehensive experiments
  python main.py --experiments

  # Generate standard dataset
  python main.py --generate-dataset --dataset-size 700 150 150
        """,
    )

    parser.add_argument("--departure", type=str, help="Departure city")
    parser.add_argument("--destination", type=str, help="Destination city")
    parser.add_argument("--date", type=str, help="Flight date (YYYY-MM-DD)")
    parser.add_argument(
        "--priority",
        type=str,
        choices=["safety", "cost", "time", "comfort", "balanced"],
        default="balanced",
        help="Flight selection priority",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Budget preference",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run full system evaluation"
    )
    parser.add_argument(
        "--compare-protocols",
        action="store_true",
        help="Run evaluation for all protocols and generate plots",
    )
    parser.add_argument(
        "--experiments", action="store_true", help="Run comprehensive experiments"
    )
    parser.add_argument(
        "--generate-dataset", action="store_true", help="Generate standard dataset"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        nargs=3,
        default=[700, 150, 150],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Dataset sizes for train/val/test",
    )

    parser.add_argument("--config-file", type=str, help="Custom configuration file")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--output-file", type=str, help="Output file for results")

    return parser


async def run_single_query(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        logger.info(
            f"Processing flight query: {args.departure} → {args.destination} on {args.date}"
        )

        preferences = {"priority": args.priority, "budget": args.budget}

        result = await process_flight_query_simple(
            departure=args.departure,
            destination=args.destination,
            date=args.date,
            preferences=preferences,
        )

        return result

    except Exception as e:
        logger.error(f"Single query failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def run_interactive_mode():
    logger.info("Starting MAMA interactive mode...")
    print("\n" + "=" * 60)
    print("MAMA Flight Selection Assistant - Interactive Mode")
    print("=" * 60)

    workflow = create_mama_workflow()
    await workflow.initialize_system()

    try:
        while True:
            print("\nEnter flight details (or 'quit' to exit):")

            departure = input("Departure city: ").strip()
            if departure.lower() == "quit":
                break

            destination = input("Destination city: ").strip()
            if destination.lower() == "quit":
                break

            date = input("Flight date (YYYY-MM-DD): ").strip()
            if date.lower() == "quit":
                break

            print("\nPreferences (press Enter for defaults):")
            priority = (
                input("Priority (safety/cost/time/comfort/balanced) [balanced]: ").strip()
                or "balanced"
            )
            budget = input("Budget (low/medium/high) [medium]: ").strip() or "medium"

            preferences = {"priority": priority, "budget": budget}

            print(f"\nProcessing query: {departure} → {destination} on {date}...")

            try:
                result = await workflow.process_flight_query(
                    departure, destination, date, preferences
                )

                print("\n" + "-" * 50)
                print("RESULTS:")
                print("-" * 50)

                if result["status"] == "success":
                    print("✓ Query processed successfully")
                    print(f"  Processing time: {result['total_processing_time']:.2f}s")
                    print(f"  Integrated score: {result['integrated_score']:.3f}")
                    print(f"  Confidence level: {result['confidence_level']:.3f}")

                    metrics = result.get("performance_metrics", {})
                    if metrics:
                        print(f"  Agents used: {metrics.get('agent_count', 0)}")
                        print(f"  Trust updates: {metrics.get('trust_updates', 0)}")

                    recommendations = result.get("final_recommendations", [])
                    if recommendations:
                        print("\nTop recommendations:")
                        for i, rec in enumerate(recommendations[:3], 1):
                            print(f"  {i}. {rec}")
                else:
                    print(f"✗ Query failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"✗ Error processing query: {e}")

            print("\n" + "=" * 60)

    finally:
        await workflow.cleanup()
        print("\nThank you for using MAMA Flight Selection Assistant!")


def print_results(result: Dict[str, Any], output_file: Optional[str] = None):
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    status = result.get("status", result.get("overall_status", "unknown"))

    if "experiment_type" in result:
        experiment_type = result.get("experiment_type", "unknown")
        if status == "success" or status == "partial_success":
            status_symbol = "✓" if status == "success" else "⚠"
            status_text = "SUCCESS" if status == "success" else "PARTIAL SUCCESS"
            print(f"{status_symbol} Status: {status_text}")
            print(f"  Experiment Type: {experiment_type}")
            print(f"  Timestamp: {result.get('timestamp', 'N/A')}")

            summary = result.get("summary", {})
            if summary:
                print("\nExperiment Summary:")
                print(f"  Total Experiments: {summary.get('total_experiments', 0)}")
                print(f"  Successful: {summary.get('successful_experiments', 0)}")
                print(f"  Overall Status: {summary.get('overall_status', 'unknown')}")

            experiments = result.get("experiments", {})
            if experiments:
                print("\nExperiment Results:")
                for exp_name, exp_result in experiments.items():
                    exp_status = exp_result.get("status", "unknown")
                    exp_symbol = "✓" if exp_status == "success" else "✗"
                    print(f"  {exp_symbol} {exp_name}: {exp_status}")
        else:
            print("✗ Status: FAILED")
            print(f"  Experiment Type: {experiment_type}")
            print(f"  Error: {result.get('error', 'Unknown error')}")

    elif result.get("evaluation_type"):
        evaluation_type = result["evaluation_type"]
        print("✓ Status: SUCCESS")
        print(f"  Evaluation Type: {evaluation_type}")
        print(f"  Timestamp: {result.get('timestamp', 'N/A')}")

        summary = result.get("summary", {})
        if summary:
            print("\nEvaluation Summary:")
            print(f"  Validation Passed: {summary.get('validation_passed', False)}")
            print(f"  Evaluation Success: {summary.get('evaluation_success', False)}")
            print(f"  Average Reward: {summary.get('average_reward', 0.0):.3f}")
            print(f"  System Ready: {summary.get('system_ready', False)}")

        scenario_eval = result.get("scenario_evaluation", {})
        aggregate = scenario_eval.get("aggregate_metrics", {})
        if aggregate:
            ndcg_k = aggregate.get('ndcg_k', 5)
            print("\nAggregate Metrics:")
            print(f"  Success Rate: {aggregate.get('success_rate', 0.0):.3f}")
            print(f"  Average MRR: {aggregate.get('average_mrr', 0.0):.3f}")
            print(f"  Average NDCG@{ndcg_k}: {aggregate.get('average_ndcg_at_k', 0.0):.3f}")
            print(f"  Average Response Time: {aggregate.get('average_response_time', 0.0):.2f}s")
            print(f"  Average Reward: {aggregate.get('average_reward', 0.0):.3f}")

        validation = result.get("system_validation", {})
        if validation:
            print("\nValidation Status:")
            print(f"  Overall Status: {validation.get('overall_status', 'unknown')}")
            component_tests = validation.get("component_tests", {})
            integration_tests = validation.get("integration_tests", {})
            performance_tests = validation.get("performance_tests", {})
            if component_tests:
                print(f"  Component Tests Passed: {component_tests.get('all_passed', False)}")
            if integration_tests:
                print(f"  Integration Tests Passed: {integration_tests.get('all_passed', False)}")
            if performance_tests:
                print(f"  Performance Tests Passed: {performance_tests.get('all_passed', False)}")

    elif 'results' in result and 'protocols' in result:
        print("✓ Status: SUCCESS")
        print("Protocol comparison summary:")
        res = result.get('results', {})
        for proto in result.get('protocols', []):
            agg = res.get(proto, {}).get('scenario_evaluation', {}).get('aggregate_metrics', {})
            avg_mrr = agg.get('average_mrr', 0.0)
            avg_art = agg.get('average_response_time', 0.0)
            print(f"  - {proto}: Average MRR = {avg_mrr:.3f}, Average ART = {avg_art:.3f}s")

    elif status == "success" or status == "passed":
        print("✓ Status: SUCCESS")
        print(f"  Query ID: {result.get('query_id', 'N/A')}")
        print(f"  Route: {result.get('departure', 'N/A')} → {result.get('destination', 'N/A')}")
        print(f"  Date: {result.get('date', 'N/A')}")
        print(f"  Processing Time: {result.get('total_processing_time', 0):.2f}s")
        print(f"  Integrated Score: {result.get('integrated_score', 0):.3f}")
        print(f"  Confidence Level: {result.get('confidence_level', 0):.3f}")

        metrics = result.get("performance_metrics", {})
        if metrics:
            print("\nPerformance Metrics:")
            print(f"  Agents Used: {metrics.get('agent_count', 0)}")
            print(f"  Trust Updates: {metrics.get('trust_updates', 0)}")
            print(f"  Phase 1 Time: {metrics.get('phase1_time', 0):.2f}s")
            print(f"  Phase 2 Time: {metrics.get('phase2_time', 0):.2f}s")
            print(f"  Phase 3 Time: {metrics.get('phase3_time', 0):.2f}s")
            print(f"  Phase 4 Time: {metrics.get('phase4_time', 0):.2f}s")

        recommendations = result.get("final_recommendations", [])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                if isinstance(rec, dict):
                    flight_id = rec.get("flight_id", f"item_{i}")
                    score = rec.get("integrated_score") or rec.get("score")
                    detail_parts = []
                    for key, label in [
                        ("weather_score", "weather"),
                        ("safety_score", "safety"),
                        ("economic_score", "economic"),
                        ("overall_economic_score", "economic"),
                        ("operational_score", "operations"),
                    ]:
                        if key in rec and isinstance(rec[key], (int, float)):
                            detail_parts.append(f"{label}={rec[key]:.3f}")
                    detail = ", ".join(detail_parts)
                    if isinstance(score, (int, float)):
                        print(
                            f"  {i}. {flight_id} (score {score:.3f}{', ' + detail if detail else ''})"
                        )
                    else:
                        print(f"  {i}. {flight_id}{' - ' + detail if detail else ''}")
                else:
                    print(f"  {i}. {rec}")
    else:
        print("✗ Status: FAILED")
        print(f"  Error: {result.get('error', 'Unknown error')}")
        print(f"  Processing Time: {result.get('total_processing_time', 0):.2f}s")

    print("=" * 60)

    if output_file:
        try:
            import json

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")


async def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = get_config()
    logger.info(f"MAMA system starting with configuration: {config.system_name}")

    try:
        if args.evaluate:
            logger.info("Running full system evaluation...")
            from scripts import run_full_evaluation

            result = await run_full_evaluation()
            print_results(result, args.output_file)

        elif args.compare_protocols:
            logger.info("Running protocol comparisons and plots...")
            from scripts import run_protocol_evaluations_and_plots
            compare_results = await run_protocol_evaluations_and_plots()
            print_results(compare_results, args.output_file)

        elif args.experiments:
            logger.info("Running comprehensive experiments...")
            from scripts import get_experiment_runner

            experiment_runner = get_experiment_runner()
            result = await experiment_runner.run_all_experiments()
            print_results(result, args.output_file)

        elif args.generate_dataset:
            logger.info("Generating standard dataset...")
            from scripts import get_dataset_generator

            dataset_generator = get_dataset_generator()
            train_size, val_size, test_size = args.dataset_size
            dataset = dataset_generator.generate_standard_dataset(
                train_size, val_size, test_size
            )
            dataset_file = "data/standard_dataset.json"
            dataset_generator.save_dataset(dataset, dataset_file)

            result = {
                "status": "success",
                "dataset_file": dataset_file,
                "train_size": train_size,
                "validation_size": val_size,
                "test_size": test_size,
                "total_queries": train_size + val_size + test_size,
            }
            print_results(result, args.output_file)

        elif args.interactive:
            await run_interactive_mode()

        elif args.departure and args.destination and args.date:
            result = await run_single_query(args)
            print_results(result, args.output_file)

        else:
            parser.print_help()
            print("\nError: Please specify a valid operation mode.")
            print(
                "Use --interactive, --evaluate, --experiments, --generate-dataset, "
                "or provide --departure, --destination, and --date"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\nApplication error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
