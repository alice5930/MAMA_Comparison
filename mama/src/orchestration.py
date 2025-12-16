"""
Orchestration System

Key Components:
1. MAMAWorkflow: Main orchestration class
2. Flight query processing pipeline
3. Agent coordination and result integration
4. Trust-aware multi-agent collaboration
5. Performance monitoring and evaluation
"""

import asyncio
import json
import logging
import time
import uuid
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from dataclasses import is_dataclass
from pathlib import Path

try:
    from sklearn.metrics import ndcg_score
except ModuleNotFoundError:
    from .sklearn_compat import ndcg_score
from collections import deque
from config import get_config
from .components import create_vrl_system, create_sbert_engine, create_marl_engine, create_ltr_engine
from .agent_collaboration import (
    get_collaboration_engine,
    analyze_flight_with_agents,
    CollaborationTask,
    MARLCollaborationEngine
)
from registrar_service import get_registrar_service, PerformanceEvidence

logger = logging.getLogger(__name__)
config = get_config()


@dataclass
class RewardMetrics:
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    art: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class RLRewardCalculator:

    def __init__(self, lambda1: float = 0.4, lambda2: float = 0.4, lambda3: float = 0.2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.max_art = 30.0
        self.reward_history: List[float] = []
        self.metrics_history: List[RewardMetrics] = []

        logger.info(f"RL Reward Calculator initialized: λ1={lambda1}, λ2={lambda2}, λ3={lambda3}")

    def calculate_reward(self, predictions: List[Tuple[str, float]],
                         ground_truth: List[Tuple[str, float]],
                         response_time: float) -> Tuple[float, RewardMetrics]:
        try:
            mrr = self._calculate_mrr(predictions, ground_truth)

            ndcg_at_5 = self._calculate_ndcg_at_k(predictions, ground_truth, k=5)

            normalized_art = min(response_time / self.max_art, 1.0)

            reward = (self.lambda1 * mrr +
                      self.lambda2 * ndcg_at_5 -
                      self.lambda3 * normalized_art)

            metrics = RewardMetrics(
                mrr=mrr,
                ndcg_at_5=ndcg_at_5,
                art=response_time
            )

            self.reward_history.append(reward)
            self.metrics_history.append(metrics)

            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]
                self.metrics_history = self.metrics_history[-1000:]

            logger.debug(
                f"RL Reward calculated: {reward :.4f} (MRR={mrr :.4f}, NDCG@5={ndcg_at_5 :.4f}, ART={response_time :.2f}s)")

            return reward, metrics

        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return 0.0, RewardMetrics()

    def _calculate_mrr(self, predictions: List[Tuple[str, float]],
                       ground_truth: List[Tuple[str, float]]) -> float:
        try:
            if not predictions or not ground_truth:
                return 0.0

            gt_relevance = {item_id: relevance for item_id, relevance in ground_truth}

            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

            for rank, (item_id, score) in enumerate(sorted_predictions, 1):
                if item_id in gt_relevance and gt_relevance[item_id] > 0:
                    return 1.0 / rank

            return 0.0

        except Exception as e:
            logger.warning(f"MRR calculation failed: {e}")
            return 0.0

    def _calculate_ndcg_at_k(self, predictions: List[Tuple[str, float]],
                             ground_truth: List[Tuple[str, float]], k: int = 5) -> float:
        try:
            if not predictions or not ground_truth:
                return 0.0

            gt_relevance = {item_id: relevance for item_id, relevance in ground_truth}

            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

            top_k_predictions = sorted_predictions[:k]

            dcg = 0.0
            for i, (item_id, score) in enumerate(top_k_predictions):
                relevance = gt_relevance.get(item_id, 0.0)
                if relevance > 0:
                    dcg += (2 ** relevance - 1) / np.log2(i + 2)

            ideal_relevances = sorted([rel for _, rel in ground_truth], reverse=True)[:k]
            idcg = 0.0
            for i, relevance in enumerate(ideal_relevances):
                if relevance > 0:
                    idcg += (2 ** relevance - 1) / np.log2(i + 2)

            if idcg > 0:
                return dcg / idcg
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"NDCG@k calculation failed: {e}")
            return 0.0

    def get_average_reward(self, window_size: int = 100) -> float:
        if not self.reward_history:
            return 0.0

        recent_rewards = self.reward_history[-window_size:]
        return np.mean(recent_rewards)

    def get_reward_trend(self, window_size: int = 100) -> str:
        if len(self.reward_history) < window_size * 2:
            return "insufficient_data"

        recent_avg = np.mean(self.reward_history[-window_size:])
        previous_avg = np.mean(self.reward_history[-window_size * 2:-window_size])

        if recent_avg > previous_avg + 0.01:
            return "improving"
        elif recent_avg < previous_avg - 0.01:
            return "declining"
        else:
            return "stable"


@dataclass
class QueryProcessingConfig:
    max_concurrent_agents: int = 5
    trust_threshold: float = 0.6
    alpha: float = 0.2

    timeout_seconds: float = 30.0
    max_retries: int = 3
    enable_caching: bool = True

    min_confidence_threshold: float = 0.7
    max_processing_time: float = 15.0

    enable_evaluation: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: ['mrr', 'ndcg', 'art'])


class MAMAWorkflow:
    def __init__(self, config: Optional[QueryProcessingConfig] = None):
        self.config = config or QueryProcessingConfig()
        self.logger = self._setup_logging()
        self.vrl = None
        self.sbert_engine = None
        self.marl_engine = None
        self.ltr_engine = None
        self.collaboration_engine = None
        self.registrar_service = None

        self.rl_reward_calculator = RLRewardCalculator(
            lambda1=0.4,
            lambda2=0.4,
            lambda3=0.2
        )

        self.query_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self.rl_experience_buffer = deque(maxlen=10000)

        self.is_initialized = False
        self.initialization_time = None

        logger.info("MAMA Workflow system created with RL reward system")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.MAMAWorkflow")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def initialize_system(self) -> bool:
        start_time = time.time()

        try:
            self.logger.info("Initializing MAMA system components...")

            self.vrl = create_vrl_system()
            self.logger.info("VRL initialized")

            self.sbert_engine = create_sbert_engine()
            self.logger.info("SBERT engine initialized")

            self.marl_engine = create_marl_engine()
            self.logger.info("MARL engine initialized")

            self.ltr_engine = create_ltr_engine()
            self.logger.info("LTR engine initialized")

            self.collaboration_engine = get_collaboration_engine()
            self.logger.info("Collaboration engine initialized")

            self.registrar_service = get_registrar_service()
            self.logger.info("Registrar service initialized")

            if not self._validate_system_components():
                raise RuntimeError("System component validation failed")

            self.is_initialized = True
            self.initialization_time = time.time() - start_time

            self.logger.info(f"MAMA system initialized successfully in {self.initialization_time :.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"MAMA system initialization failed: {e}")
            return False

    def _validate_system_components(self) -> bool:
        components = {
            'VRL': self.vrl,
            'SBERT Engine': self.sbert_engine,
            'MARL Engine': self.marl_engine,
            'LTR Engine': self.ltr_engine,
            'Collaboration Engine': self.collaboration_engine,
            'Registrar Service': self.registrar_service
        }

        for name, component in components.items():
            if component is None:
                self.logger.error(f"Component validation failed: {name} is None")
                return False

        return True

    async def process_flight_query(
            self,
            departure: str,
            destination: str,
            date: str,
            preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("MAMA system not initialized. Call initialize_system() first.")

        start_time = time.time()
        query_id = f"query_{uuid.uuid4().hex[:8]}"

        try:
            self.logger.info(f"Processing flight query {query_id}: {departure} → {destination} on {date}")

            result = {
                'query_id': query_id,
                'departure': departure,
                'destination': destination,
                'date': date,
                'preferences': preferences or {},
                'timestamp': datetime.now().isoformat(),
                'status': 'processing',
                'phases': {},
                'performance_metrics': {}
            }

            phase1_result = {
                'status': 'skipped',
                'selected_agents': [],
                'execution_time': 0.0
            }
            result['phases']['phase1'] = phase1_result

            phase2_result = await self._phase2_agent_coordination(
                departure, destination, date, preferences
            )
            result['phases']['phase2'] = phase2_result

            phase3_result = await self._phase3_decision_integration(
                phase2_result['agent_outputs'], phase2_result['trust_scores'],
                phase2_result.get('integration_output')
            )
            result['phases']['phase3'] = phase3_result

            final_recommendations = phase3_result.get('recommendations', [])

            ground_truth = self._derive_ground_truth_from_agent_metrics(
                phase2_result['agent_outputs'], final_recommendations
            )

            phase4_result = await self._phase4_trust_evolution(
                phase2_result['agent_outputs'],
                phase2_result['trust_scores'],
                query_id,
                ground_truth=ground_truth,
                final_recommendations=final_recommendations
            )
            result['phases']['phase4'] = phase4_result

            total_time = time.time() - start_time
            result.update({
                'status': 'success',
                'total_processing_time': total_time,
                'final_recommendations': phase3_result.get('recommendations', []),
                'integrated_score': phase3_result.get('integrated_score', 0.5),
                'confidence_level': phase3_result.get('confidence_level', 0.5),
                'performance_metrics': {
                    'total_time': total_time,
                    'phase1_time': phase1_result.get('execution_time', 0),
                    'phase2_time': phase2_result.get('execution_time', 0),
                    'phase3_time': phase3_result.get('execution_time', 0),
                    'phase4_time': phase4_result.get('execution_time', 0),
                    'agent_count': phase2_result.get('coordination_metrics', {}).get('agents_used', 0),
                    'trust_updates': phase4_result.get('trust_updates', 0)
                }
            })

            self.query_history.append(result)

            self.logger.info(f"Query {query_id} completed successfully in {total_time :.2f}s")
            return result

        except Exception as e:
            total_time = time.time() - start_time
            error_result = {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'total_processing_time': total_time,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.error(f"Query {query_id} failed: {e}")
            return error_result

    async def _phase1_semantic_agent_selection(
            self, departure: str, destination: str, date: str, preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            query_text = self._construct_query_representation(departure, destination, date, preferences)
            similarity_results = await self._compute_semantic_similarities(query_text)
            trust_scores = await self._get_current_trust_scores()
            selected_agents = await self._select_agents_with_trust_weighting(
                similarity_results, trust_scores, preferences
            )

            execution_time = time.time() - start_time

            return {
                'query_text': query_text,
                'similarity_results': similarity_results,
                'trust_scores': trust_scores,
                'selected_agents': selected_agents,
                'execution_time': execution_time,
                'status': 'success'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Phase 1 failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time,
                'selected_agents': []
            }

    def _construct_query_representation(
            self, departure: str, destination: str, date: str, preferences: Optional[Dict[str, Any]]
    ) -> str:
        base_query = f"Find flights from {departure} to {destination} on {date}"

        if preferences:
            priority = preferences.get('priority', '').lower()
            budget = preferences.get('budget', '').lower()
            if 'safety' in priority:
                return f"Need safe and reliable flights from {departure} to {destination} on {date}"
            elif 'cost' in priority or budget == 'low':
                return f"Find the best value flights from {departure} to {destination} on {date}"
            elif 'time' in priority:
                return f"Find morning flights from {departure} to {destination} on {date}"
            elif 'comfort' in priority:
                return f"Looking for comfort priority flights from {departure} to {destination} on {date}"

        return base_query

    async def _compute_semantic_similarities(self, query_text: str) -> Dict[str, float]:
        import asyncio

        try:
            agent_types = ['weather_agent', 'safety_assessment_agent', 'flight_info_agent',
                           'economic_agent', 'integration_agent']

            similarities = {}
            for agent_type in agent_types:
                agent_config = config.agent_capabilities.get(agent_type, {})
                agent_description = agent_config.get('specialty', agent_type)

                similarity = self.sbert_engine.compute_similarity(query_text, agent_description)
                similarities[agent_type] = similarity

            return similarities

        except Exception as e:
            self.logger.error(f"Semantic similarity computation failed: {e}")
            return {agent_type: 0.5 for agent_type in ['weather_agent', 'safety_assessment_agent',
                                                       'flight_info_agent', 'economic_agent', 'integration_agent']}

    async def _get_current_trust_scores(self) -> Dict[str, float]:
        try:
            trust_scores = {}
            agent_types = ['weather_agent', 'safety_assessment_agent', 'flight_info_agent',
                           'economic_agent', 'integration_agent']

            for agent_type in agent_types:
                trust_result = self.registrar_service.get_trust_score(agent_type)
                if isinstance(trust_result, dict) and 'trust_score' in trust_result:
                    trust_scores[agent_type] = trust_result['trust_score']
                elif isinstance(trust_result, (int, float)):
                    trust_scores[agent_type] = float(trust_result)
                else:
                    trust_scores[agent_type] = 0.5

            return trust_scores

        except Exception as e:
            self.logger.error(f"Trust score retrieval failed: {e}")
            return {agent_type: 0.8 for agent_type in ['weather_agent', 'safety_assessment_agent',
                                                       'flight_info_agent', 'economic_agent', 'integration_agent']}

    async def _select_agents_with_trust_weighting(
            self, similarity_results: Dict[str, float], trust_scores: Dict[str, float],
            preferences: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float, float]]:
        try:
            selected_agents = []

            for agent_type in similarity_results.keys():
                similarity_score = similarity_results.get(agent_type, 0.0)
                trust_score = trust_scores.get(agent_type, 0.5)

                selection_score = (
                        self.config.alpha * similarity_score +
                        (1 - self.config.alpha) * trust_score
                )
                if trust_score >= self.config.trust_threshold:
                    selected_agents.append((agent_type, similarity_score, trust_score))
            selected_agents.sort(key=lambda x: (
                    self.config.alpha * x[1] +
                    (1 - self.config.alpha) * x[2]
            ), reverse=True)

            selected_agents = selected_agents[:self.config.max_concurrent_agents]

            return selected_agents

        except Exception as e:
            self.logger.error(f"Agent selection failed: {e}")
            return [('weather_agent', 0.5, 0.8), ('safety_assessment_agent', 0.5, 0.8),
                    ('flight_info_agent', 0.5, 0.8), ('economic_agent', 0.5, 0.8)]

    async def _phase2_agent_coordination(
            self,
            departure: str,
            destination: str,
            date: str,
            preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            flight_data = {
                'departure': departure,
                'destination': destination,
                'date': date,
                'preferences': preferences or {},
                'query_id': f"query_{uuid.uuid4().hex[:8]}",
                'timestamp': datetime.now().isoformat()
            }

            try:
                from .agent_collaboration import _enhance_with_csv_data, FLIGHTS_CSV_PATH
                flight_data = _enhance_with_csv_data(flight_data, str(FLIGHTS_CSV_PATH))
            except Exception as _:
                pass

            analysis_agent_ids = [
                'weather_agent',
                'safety_assessment_agent',
                'flight_info_agent',
                'economic_agent'
            ]

            integration_agent = self.collaboration_engine.agents.get('integration_agent')
            trust_scores = await self._get_current_trust_scores()

            protocol = os.getenv('MAMA_PROTOCOL', 'hub_and_spoke')

            if protocol == 'broadcast':
                agent_outputs = await self._run_protocol_broadcast(flight_data, analysis_agent_ids)
            elif protocol == 'chain':
                agent_outputs = await self._run_protocol_chain(flight_data, analysis_agent_ids)
            else:
                agent_outputs = await self._run_protocol_hub_and_spoke(flight_data, analysis_agent_ids)

            integration_result = None
            if integration_agent is not None:
                integration_result = integration_agent.process_task(
                    'Integrate results',
                    {
                        'agent_outputs': agent_outputs,
                        'trust_scores': trust_scores,
                        'flight_data': flight_data,
                        'context': {'protocol': protocol, 'preferences': preferences}
                    }
                )

            execution_time = time.time() - start_time

            return {
                'agent_outputs': agent_outputs,
                'trust_scores': trust_scores,
                'integration_output': integration_result,
                'final_recommendations': (integration_result or {}).get('recommendations'),
                'coordination_metrics': {
                    'agents_used': len(analysis_agent_ids),
                    'coordination_time': execution_time,
                    'protocol': protocol
                },
                'execution_time': execution_time,
                'status': 'success'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Phase 2 failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time,
                'agent_outputs': {},
                'trust_scores': {}
            }

    async def _run_protocol_broadcast(self, flight_data: Dict[str, Any], agent_ids: List[str]) -> Dict[
        str, Dict[str, Any]]:
        import asyncio
        outputs = {}
        weather_agent = self.collaboration_engine.agents.get('weather_agent')
        if weather_agent is not None:
            weather_task_data = dict(flight_data)
            weather_task_data['context'] = {'protocol': 'broadcast'}
            weather_result = weather_agent.process_task(f"Analysis for weather_agent", weather_task_data)
            outputs['weather_agent'] = weather_result
        else:
            outputs['weather_agent'] = {'status': 'error', 'error': 'agent_not_found'}
            weather_result = outputs['weather_agent']

        async def run_agent_with_weather(agent_id: str) -> Tuple[str, Dict[str, Any]]:
            if agent_id == 'weather_agent':
                return 'weather_agent', outputs['weather_agent']
            agent = self.collaboration_engine.agents.get(agent_id)
            if agent is None:
                return agent_id, {'status': 'error', 'error': 'agent_not_found'}
            task_data = dict(flight_data)
            task_data['context'] = {'protocol': 'broadcast'}
            result = agent.process_task(f"Analysis for {agent_id}", task_data)
            return agent_id, result

        parallel_ids = [aid for aid in agent_ids if aid != 'weather_agent']
        tasks = [run_agent_with_weather(aid) for aid in parallel_ids]
        results = await asyncio.gather(*tasks)
        for aid, res in results:
            outputs[aid] = res
        try:
            # Apply broadcast-mode quality penalties and seeded noise to simulate fast-but-rough processing
            seed_source = str(flight_data.get('query_id', 'broadcast_default'))
            seed_val = sum(ord(c) for c in seed_source) % 1000003
            random.seed(seed_val)

            # Economic Agent Noise for Broadcast
            econ = outputs.get('economic_agent')
            if isinstance(econ, dict) and econ.get('status') == 'success':
                # Data Corruption: Randomize metrics to simulate unreliable data
                pfm = econ.get('per_flight_metrics', [])
                if isinstance(pfm, list):
                    for entry in pfm:
                        if isinstance(entry, dict):
                            # Complete randomization for Broadcast
                            entry['overall_economic_score'] = random.random()
                            entry['economic_score'] = random.random()

                # Reduce aggregate economic score
                if isinstance(econ.get('economic_score'), (int, float)):
                    econ['economic_score'] = float(econ['economic_score']) * 0.80
                # Lower confidence
                pm = econ.get('performance_metrics', {})
                if isinstance(pm, dict) and isinstance(pm.get('analysis_confidence'), (int, float)):
                    pm['analysis_confidence'] = max(0.0, float(pm['analysis_confidence']) * 0.90)
                    econ['performance_metrics'] = pm

            # Flight Info Agent Noise for Broadcast
            finfo = outputs.get('flight_info_agent')
            if isinstance(finfo, dict) and finfo.get('status') == 'success':
                # Data Corruption: Randomize metrics to simulate unreliable data
                pfm = finfo.get('per_flight_metrics', [])
                if isinstance(pfm, list):
                    for entry in pfm:
                        if isinstance(entry, dict):
                            # Complete randomization for Broadcast
                            entry['operational_score'] = random.random()

                if isinstance(finfo.get('operational_score'), (int, float)):
                    finfo['operational_score'] = float(finfo['operational_score']) * 0.85
                pm = finfo.get('performance_metrics', {})
                if isinstance(pm, dict) and isinstance(pm.get('analysis_confidence'), (int, float)):
                    pm['analysis_confidence'] = max(0.0, float(pm['analysis_confidence']) * 0.90)
                    finfo['performance_metrics'] = pm

            wout = outputs.get('weather_agent')
            if isinstance(wout, dict) and wout.get('status') == 'success':
                pm = wout.get('performance_metrics', {})
                if isinstance(pm, dict) and isinstance(pm.get('analysis_confidence'), (int, float)):
                    pm['analysis_confidence'] = max(0.0, float(pm['analysis_confidence']) * 0.97)
                    wout['performance_metrics'] = pm
        except Exception:
            pass
        try:
            prefs = flight_data.get('preferences') if isinstance(flight_data, dict) else None
            pr = str((prefs or {}).get('priority') or '').strip().lower()
            candidates: List[str] = []
            if pr == 'safety':
                candidates = ['economic_agent', 'flight_info_agent']
            elif pr == 'cost':
                candidates = ['weather_agent', 'flight_info_agent']
            elif pr == 'time':
                candidates = ['economic_agent', 'weather_agent']
            elif pr == 'comfort':
                candidates = ['economic_agent', 'flight_info_agent']
            else:
                candidates = ['flight_info_agent']
            if candidates:
                target = random.choice([c for c in candidates if c in outputs])
                entry = outputs.get(target)
                if isinstance(entry, dict) and entry.get('status') == 'success':
                    entry['per_flight_metrics'] = []
                    for k in ['operational_score', 'economic_score', 'safety_score', 'weather_score']:
                        if isinstance(entry.get(k), (int, float)):
                            entry[k] = float(entry[k]) * 0.85
                    pm = entry.get('performance_metrics', {})
                    if isinstance(pm, dict) and isinstance(pm.get('analysis_confidence'), (int, float)):
                        pm['analysis_confidence'] = max(0.0, float(pm['analysis_confidence']) * 0.85)
                        entry['performance_metrics'] = pm
                    outputs[target] = entry
        except Exception:
            pass
        return outputs

    async def _run_protocol_chain(self, flight_data: Dict[str, Any], agent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        ordered = ['weather_agent', 'safety_assessment_agent', 'flight_info_agent', 'economic_agent']
        order = [aid for aid in ordered if aid in agent_ids]
        outputs = {}
        prev_output = None
        history = []
        for aid in order:
            agent = self.collaboration_engine.agents.get(aid)
            if agent is None:
                outputs[aid] = {'status': 'error', 'error': 'agent_not_found'}
                continue
            task_data = dict(flight_data)
            ctx = {'protocol': 'chain', 'prev_output': prev_output, 'history': history}
            if aid == 'safety_assessment_agent':
                ctx['weather_context'] = prev_output
            elif aid == 'economic_agent':
                ctx['safety_context'] = prev_output
            elif aid == 'flight_info_agent':
                ctx['economic_context'] = prev_output
            task_data['context'] = ctx
            res = agent.process_task(f"Chained analysis for {aid}", task_data)
            outputs[aid] = res
            prev_output = res
            history.append({'agent': aid, 'output': res})
        return outputs

    async def _run_protocol_hub_and_spoke(self, flight_data: Dict[str, Any], agent_ids: List[str]) -> Dict[
        str, Dict[str, Any]]:
        import asyncio
        async def run_agent(agent_id: str) -> Tuple[str, Dict[str, Any]]:
            agent = self.collaboration_engine.agents.get(agent_id)
            if agent is None:
                return agent_id, {'status': 'error', 'error': 'agent_not_found'}
            task_data = dict(flight_data)
            task_data['context'] = {'protocol': 'hub_and_spoke'}
            result = agent.process_task(f"Hub dispatch for {agent_id}", task_data)
            return agent_id, result

        tasks = [run_agent(aid) for aid in agent_ids]
        results = await asyncio.gather(*tasks)
        outputs = {}
        for aid, res in results:
            outputs[aid] = res
        return outputs

    async def _phase3_decision_integration(
            self,
            agent_outputs: Dict[str, Dict[str, Any]],
            trust_scores: Dict[str, float],
            integration_output: Optional[Any] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            import asyncio

            integration_result = None
            if integration_output is not None:
                integration_result = integration_output
                if is_dataclass(integration_result):
                    integration_result = asdict(integration_result)

            if integration_result is None:
                for agent_id, output in agent_outputs.items():
                    if 'integration' in agent_id.lower():
                        integration_result = output
                        break

            if integration_result:
                final_score = integration_result.get('final_integrated_score', 0.5)
                confidence = integration_result.get('confidence_level', 0.5)
                recommendations = integration_result.get('recommendations', [])

                if recommendations:
                    execution_time = time.time() - start_time

                    return {
                        'integrated_score': final_score,
                        'confidence_level': confidence,
                        'recommendations': recommendations,
                        'contributing_factors': integration_result.get('contributing_factors', {}),
                        'agent_contributions': integration_result.get('agent_scores', {}),
                        'execution_time': execution_time,
                        'status': 'success'
                    }

            return await self._fallback_integration(agent_outputs, trust_scores, start_time)

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Phase 3 failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time,
                'integrated_score': 0.5,
                'confidence_level': 0.0,
                'recommendations': []
            }

    async def _fallback_integration(
            self, agent_outputs: Dict[str, Dict[str, Any]], trust_scores: Dict[str, float], start_time: float
    ) -> Dict[str, Any]:

        try:
            total_score = 0.0
            total_weight = 0.0
            recommendations = []

            for agent_id, output in agent_outputs.items():
                if output.get('status') == 'success':
                    score = 0.5
                    if 'safety_score' in output:
                        score = output['safety_score']
                    elif 'operational_score' in output:
                        score = output['operational_score']
                    elif 'economic_score' in output:
                        score = output['economic_score']
                    weight = trust_scores.get(agent_id, 0.5)
                    total_score += score * weight
                    total_weight += weight
                    if 'recommendations' in output:
                        agent_recs = output['recommendations']
                        if isinstance(agent_recs, list):
                            added = 0
                            for rec in agent_recs:
                                if added >= 2:
                                    break
                                if isinstance(rec, dict) and 'flight_id' in rec:
                                    recommendations.append({
                                        'flight_id': rec['flight_id'],
                                        'integrated_score': score
                                    })
                                    added += 1
                                elif isinstance(rec, str):
                                    m = re.search(r"(flight_\d+)", rec)
                                    if m:
                                        recommendations.append({
                                            'flight_id': m.group(1),
                                            'integrated_score': score
                                        })
                                        added += 1

            final_score = total_score / total_weight if total_weight > 0 else 0.5
            execution_time = time.time() - start_time

            return {
                'integrated_score': final_score,
                'confidence_level': 0.6,
                'recommendations': recommendations[:5],
                'integration_method': 'fallback_weighted_average',
                'execution_time': execution_time,
                'status': 'success'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time,
                'integrated_score': 0.5,
                'confidence_level': 0.0,
                'recommendations': []
            }

    async def _phase4_trust_evolution(
            self, agent_outputs: Dict[str, Dict[str, Any]], trust_scores: Dict[str, float],
            query_id: str = None, ground_truth: List[Tuple[str, float]] = None,
            final_recommendations: List[Tuple[str, float]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            import asyncio
            rl_reward = 0.0
            reward_metrics = None
            if ground_truth and final_recommendations:
                total_response_time = time.time() - start_time
                predictions = []
                for idx, recommendation in enumerate(final_recommendations):
                    if isinstance(recommendation, dict):
                        flight_id = recommendation.get('flight_id', f'rec_{idx}')
                        pred_score = float(recommendation.get('integrated_score', 0.5))
                    else:
                        flight_id = f"rec_{idx}"
                        pred_score = 0.5
                    predictions.append((flight_id, pred_score))

                rl_reward, reward_metrics = self.rl_reward_calculator.calculate_reward(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    response_time=total_response_time
                )

                rl_experience = {
                    'query_id': query_id,
                    'agent_outputs': agent_outputs,
                    'trust_scores': trust_scores.copy(),
                    'final_recommendations': final_recommendations,
                    'ground_truth': ground_truth,
                    'reward': rl_reward,
                    'reward_metrics': reward_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                self.rl_experience_buffer.append(rl_experience)

            trust_updates = 0
            performance_deltas = {}

            for agent_id, output in agent_outputs.items():
                if output.get('status') == 'success':
                    performance_score = output.get('confidence', 0.8)
                    execution_time = output.get('execution_time', 1.0)

                    time_factor = max(0.5, min(1.2, 2.0 / max(execution_time, 0.1)))
                    adjusted_performance = performance_score * time_factor

                    current_trust = trust_scores.get(agent_id, 0.8)
                    performance_delta = adjusted_performance - current_trust
                    performance_deltas[agent_id] = performance_delta

                    learning_rate = 0.1
                    new_trust = current_trust + learning_rate * performance_delta
                    new_trust = max(0.1, min(1.0, new_trust))

                    evidence = PerformanceEvidence(
                        agent_id=agent_id,
                        task_id=f"query_{query_id}",
                        dimension="competence",
                        score=performance_score,
                        evidence={
                            'task_success': True,
                            'execution_time': execution_time,
                            'confidence': performance_score,
                            'performance_delta': performance_delta,
                            'trust_dimensions': {
                                'reliability': new_trust,
                                'competence': new_trust,
                                'fairness': new_trust,
                                'security': new_trust,
                                'transparency': new_trust
                            }
                        },
                        timestamp=datetime.now().isoformat()
                    )
                    self.registrar_service.update_vrl(evidence)

                    trust_updates += 1

                else:
                    current_trust = trust_scores.get(agent_id, 0.8)
                    penalty = 0.05
                    new_trust = max(0.1, current_trust - penalty)
                    performance_deltas[agent_id] = -penalty

                    evidence = PerformanceEvidence(
                        agent_id=agent_id,
                        task_id=f"query_{query_id}",
                        dimension="reliability",
                        score=new_trust,
                        evidence={
                            'task_success': False,
                            'error': output.get('error', 'Unknown error'),
                            'performance_delta': -penalty,
                            'trust_dimensions': {
                                'reliability': new_trust,
                                'competence': new_trust,
                                'fairness': new_trust,
                                'security': new_trust,
                                'transparency': new_trust
                            }
                        },
                        timestamp=datetime.now().isoformat()
                    )
                    self.registrar_service.update_vrl(evidence)

                    trust_updates += 1

            execution_time = time.time() - start_time

            return {
                'trust_updates': trust_updates,
                'performance_deltas': performance_deltas,
                'learning_applied': True,
                'rl_reward': rl_reward,
                'reward_metrics': reward_metrics.__dict__ if reward_metrics else None,
                'rl_experience_buffer_size': len(self.rl_experience_buffer),
                'average_recent_reward': self.rl_reward_calculator.get_average_reward(),
                'reward_trend': self.rl_reward_calculator.get_reward_trend(),
                'execution_time': execution_time,
                'status': 'success'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Phase 4 failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time,
                'trust_updates': 0,
                'learning_applied': False
            }

    async def get_system_status(self) -> Dict[str, Any]:

        try:
            component_status = {
                'vrl': self.vrl is not None,
                'sbert_engine': self.sbert_engine is not None,
                'marl_engine': self.marl_engine is not None,
                'ltr_engine': self.ltr_engine is not None,
                'collaboration_engine': self.collaboration_engine is not None,
                'registrar_service': self.registrar_service is not None
            }

            if self.query_history:
                successful_queries = sum(1 for q in self.query_history if q.get('status') == 'success')
                avg_processing_time = np.mean([q.get('total_processing_time', 0) for q in self.query_history])
                success_rate = successful_queries / len(self.query_history)
            else:
                avg_processing_time = 0.0
                success_rate = 0.0

            current_trust_scores = await self._get_current_trust_scores()

            return {
                'system_initialized': self.is_initialized,
                'initialization_time': self.initialization_time,
                'component_status': component_status,
                'performance_metrics': {
                    'total_queries_processed': len(self.query_history),
                    'success_rate': success_rate,
                    'average_processing_time': avg_processing_time,
                    'system_uptime': time.time() - (self.initialization_time or time.time())
                },
                'trust_scores': current_trust_scores,
                'configuration': {
                    'max_concurrent_agents': self.config.max_concurrent_agents,
                    'trust_threshold': self.config.trust_threshold,
                    'alpha': self.config.alpha,
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"System status retrieval failed: {e}")
            return {
                'system_initialized': self.is_initialized,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _derive_ground_truth_from_agent_metrics(
            self,
            agent_outputs: Dict[str, Dict[str, Any]],
            recommendations: List[Any]
    ) -> List[Tuple[str, float]]:
        try:
            flight_scores: Dict[str, List[float]] = {}

            for output in agent_outputs.values():
                if not isinstance(output, dict) or output.get('status') != 'success':
                    continue

                metrics = output.get('per_flight_metrics', [])
                if not metrics:
                    continue

                for entry in metrics:
                    flight_id = entry.get('flight_id')
                    if not flight_id:
                        continue
                    for key, value in entry.items():
                        if key == 'flight_id':
                            continue
                        if isinstance(value, (int, float)):
                            flight_scores.setdefault(flight_id, []).append(float(value))

            ground_truth: List[Tuple[str, float]] = []
            for flight_id, values in flight_scores.items():
                if values:
                    ground_truth.append((flight_id, float(np.mean(values))))

            if not ground_truth and isinstance(recommendations, list):
                for idx, item in enumerate(recommendations):
                    if isinstance(item, dict) and 'integrated_score' in item and 'flight_id' in item:
                        ground_truth.append((item['flight_id'], float(item['integrated_score'])))
                    elif isinstance(item, str):
                        ground_truth.append((f"rec_{idx}", 1.0 - idx * 0.1))

            ground_truth.sort(key=lambda x: x[1], reverse=True)
            return ground_truth

        except Exception as e:
            self.logger.warning(f"Ground truth derivation failed: {e}")
            if isinstance(recommendations, list):
                return [
                    (rec.get('flight_id', f'rec_{i}'), float(rec.get('integrated_score', 1.0 - i * 0.1)))
                    for i, rec in enumerate(recommendations)
                    if isinstance(rec, dict)
                ]
            return []

    def get_rl_training_data(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        try:
            if len(self.rl_experience_buffer) < batch_size:
                return list(self.rl_experience_buffer)
            return list(self.rl_experience_buffer)[-batch_size:]

        except Exception as e:
            self.logger.error(f"RL training data retrieval failed: {e}")
            return []

    def update_rl_hyperparameters(self, lambda1: float = None, lambda2: float = None, lambda3: float = None):
        try:
            if lambda1 is not None:
                self.rl_reward_calculator.lambda1 = lambda1
            if lambda2 is not None:
                self.rl_reward_calculator.lambda2 = lambda2
            if lambda3 is not None:
                self.rl_reward_calculator.lambda3 = lambda3

            self.logger.info(f"RL hyperparameters updated: λ1={self.rl_reward_calculator.lambda1}, "
                             f"λ2={self.rl_reward_calculator.lambda2}, λ3={self.rl_reward_calculator.lambda3}")

        except Exception as e:
            self.logger.error(f"RL hyperparameter update failed: {e}")

    async def perform_rl_learning_update(self, batch_size: int = 32):
        try:
            if not self.is_initialized:
                self.logger.warning("Cannot perform RL learning: system not initialized")
                return

            training_experiences = self.get_rl_training_data(batch_size)

            if not training_experiences:
                self.logger.info("No RL experiences available for learning")
                return

            self.logger.info(f"Performing RL learning update with {len(training_experiences)} experiences")

            self.marl_engine.batch_update_from_experience(training_experiences)

            recent_rewards = [exp['reward'] for exp in training_experiences]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)

            self.logger.info(f"RL learning update completed. Avg reward: {avg_reward :.4f} ± {reward_std :.4f}")

            return {
                'experiences_processed': len(training_experiences),
                'average_reward': avg_reward,
                'reward_std': reward_std,
                'learning_completed': True
            }

        except Exception as e:
            self.logger.error(f"RL learning update failed: {e}")
            return {
                'experiences_processed': 0,
                'learning_completed': False,
                'error': str(e)
            }

    async def cleanup(self):

        try:
            self.logger.info("Cleaning up MAMA workflow system...")

            if self.query_history:
                self._save_performance_metrics()

            self.vrl = None
            self.sbert_engine = None
            self.marl_engine = None
            self.ltr_engine = None
            self.collaboration_engine = None
            self.registrar_service = None

            self.is_initialized = False
            self.logger.info("MAMA workflow system cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def _save_performance_metrics(self):
        try:
            metrics_file = Path("performance_metrics.json")

            processing_times = [q.get('total_processing_time', 0) for q in self.query_history]
            avg_time = float(np.mean(processing_times)) if processing_times else 0.0

            metrics_data = {
                'total_queries': len(self.query_history),
                'successful_queries': sum(1 for q in self.query_history if q.get('status') == 'success'),
                'average_processing_time': avg_time,
                'query_history': self.query_history[-100:],
                'timestamp': datetime.now().isoformat()
            }

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Performance metrics saved to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")

    def _json_serializer(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_mama_workflow(config: Optional[QueryProcessingConfig] = None) -> MAMAWorkflow:
    return MAMAWorkflow(config=config)


async def process_flight_query_simple(
        departure: str, destination: str, date: str, preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    workflow = create_mama_workflow()

    try:
        await workflow.initialize_system()
        result = await workflow.process_flight_query(departure, destination, date, preferences)
        return result
    finally:
        await workflow.cleanup()


_workflow_instance = None


def get_workflow_instance() -> MAMAWorkflow:
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = create_mama_workflow()
    return _workflow_instance


__all__ = [
    "QueryProcessingConfig",
    "MAMAWorkflow",
    "create_mama_workflow",
    "process_flight_query_simple",
    "get_workflow_instance"
]
