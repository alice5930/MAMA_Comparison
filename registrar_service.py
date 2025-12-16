
"""
Registrar Service
"""

import json 
import logging 
import hashlib 
import threading 
import time 
from typing import Dict ,Any ,List ,Optional 
from datetime import datetime 
from dataclasses import dataclass ,asdict 
from pathlib import Path 
import os 
import sys 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from src .components import (
VRL ,
TrustDimension ,
TrustRecord 
)

logger =logging .getLogger (__name__ )

@dataclass 
class PerformanceEvidence :
    agent_id :str 
    task_id :str 
    dimension :str 
    score :float 
    evidence :Dict [str ,Any ]
    timestamp :str 
    evaluator :str ="agent_self_report"

class MAMARegistrarService :

    def __init__ (self ,data_dir :str ="data/registrar"):
        self .data_dir =Path (data_dir )
        self .data_dir .mkdir (parents =True ,exist_ok =True )

        self .vrl =VRL ()

        self .lock =threading .RLock ()

        self .service_id =f"registrar_{int (time .time ())}"
        self .startup_time =datetime .now ()

        self ._load_vrl_data ()

        logger .info (f"MAMA Registrar Service initialized: {self .service_id }")

    def _load_vrl_data (self ):
        try :
            vrl_file =self .data_dir /"vrl_records.json"
            if vrl_file .exists ():
                with open (vrl_file ,'r')as f :
                    data =json .load (f )

                for record_data in data .get ('records',[]):
                    record =TrustRecord (
                    agent_id =record_data ['agent_id'],
                    timestamp =datetime .fromisoformat (record_data ['timestamp']),
                    dimension =TrustDimension (record_data ['dimension']),
                    score =record_data ['score'],
                    evidence =record_data ['evidence'],
                    evaluator =record_data ['evaluator'],
                    transaction_hash =record_data ['transaction_hash'],
                    previous_hash =record_data .get ('previous_hash',''),
                    block_index =record_data .get ('block_index',0 )
                    )
                    self .vrl .trust_records .append (record )

                logger .info (f"Loaded {len (self .vrl .trust_records )} trust records from storage")

                integrity_check =self .vrl .verify_hash_chain_integrity ()
                if not integrity_check ['valid']:
                    logger .warning (f"Hash chain integrity issues detected: {integrity_check ['message']}")
                else :
                    logger .info ("Hash chain integrity verified successfully")

        except Exception as e :
            logger .error (f"Failed to load VRL data: {e }")

    def _save_vrl_data (self ):
        try :
            vrl_file =self .data_dir /"vrl_records.json"

            records_data =[]
            for record in self .vrl .trust_records :
                record_dict ={
                'agent_id':record .agent_id ,
                'timestamp':record .timestamp .isoformat (),
                'dimension':record .dimension .value ,
                'score':record .score ,
                'evidence':record .evidence ,
                'evaluator':record .evaluator ,
                'transaction_hash':record .transaction_hash ,
                'previous_hash':record .previous_hash ,
                'block_index':record .block_index 
                }
                records_data .append (record_dict )

            data ={
            'service_id':self .service_id ,
            'last_updated':datetime .now ().isoformat (),
            'total_records':len (records_data ),
            'records':records_data 
            }

            with open (vrl_file ,'w')as f :
                json .dump (data ,f ,indent =2 )

            logger .debug (f"Saved {len (records_data )} trust records to storage")

        except Exception as e :
            logger .error (f"Failed to save VRL data: {e }")

    def update_vrl (self ,evidence :PerformanceEvidence )->Dict [str ,Any ]:
        with self .lock :
            try :
                if not self ._validate_evidence (evidence ):
                    return {
                    'success':False ,
                    'error':'Invalid performance evidence',
                    'transaction_hash':None 
                    }

                try :
                    dimension =TrustDimension (evidence .dimension .lower ())
                except ValueError :
                    return {
                    'success':False ,
                    'error':f'Invalid trust dimension: {evidence .dimension }',
                    'transaction_hash':None 
                    }

                transaction_hash =self .vrl .record_trust_evaluation (
                agent_id =evidence .agent_id ,
                dimension =dimension ,
                score =evidence .score ,
                evidence =evidence .evidence ,
                evaluator =evidence .evaluator 
                )

                self ._save_vrl_data ()

                logger .info (f"VRL updated for agent {evidence .agent_id }: {evidence .dimension } = {evidence .score }")

                return {
                'success':True ,
                'transaction_hash':transaction_hash ,
                'block_index':len (self .vrl .trust_records )-1 ,
                'timestamp':datetime .now ().isoformat ()
                }

            except Exception as e :
                logger .error (f"Failed to update VRL: {e }")
                return {
                'success':False ,
                'error':str (e ),
                'transaction_hash':None 
                }

    def _validate_evidence (self ,evidence :PerformanceEvidence )->bool :
        try :
            if not evidence .agent_id or not evidence .dimension :
                return False 

            if not 0.0 <=evidence .score <=1.0 :
                return False 

            valid_dimensions =[dim .value for dim in TrustDimension ]
            if evidence .dimension .lower ()not in valid_dimensions :
                return False 

            return True 

        except Exception as e :
            logger .error (f"Evidence validation error: {e }")
            return False 

    def get_trust_score (self ,agent_id :str )->Dict [str ,Any ]:
        with self .lock :
            try :
                trust_summary =self .vrl .calculate_overall_trust_score (agent_id )
                return {
                'success':True ,
                'agent_id':agent_id ,
                'trust_summary':trust_summary ,
                'timestamp':datetime .now ().isoformat ()
                }

            except Exception as e :
                logger .error (f"Failed to get trust score for {agent_id }: {e }")
                return {
                'success':False ,
                'error':str (e ),
                'agent_id':agent_id 
                }

    def verify_integrity (self )->Dict [str ,Any ]:
        with self .lock :
            try :
                integrity_result =self .vrl .verify_hash_chain_integrity ()
                return {
                'success':True ,
                'integrity_check':integrity_result ,
                'service_id':self .service_id ,
                'timestamp':datetime .now ().isoformat ()
                }

            except Exception as e :
                logger .error (f"Integrity verification failed: {e }")
                return {
                'success':False ,
                'error':str (e )
                }

    def get_service_status (self )->Dict [str ,Any ]:
        with self .lock :
            return {
            'service_id':self .service_id ,
            'startup_time':self .startup_time .isoformat (),
            'total_records':len (self .vrl .trust_records ),
            'data_directory':str (self .data_dir ),
            'status':'active',
            'timestamp':datetime .now ().isoformat ()
            }

_registrar_service =None 

def get_registrar_service ()->MAMARegistrarService :
    global _registrar_service 
    if _registrar_service is None :
        _registrar_service =MAMARegistrarService ()
    return _registrar_service 

def update_agent_trust (agent_id :str ,dimension :str ,score :float ,
evidence :Dict [str ,Any ],task_id :str =None )->Dict [str ,Any ]:
    registrar =get_registrar_service ()

    performance_evidence =PerformanceEvidence (
    agent_id =agent_id ,
    task_id =task_id or f"task_{int (time .time ())}",
    dimension =dimension ,
    score =score ,
    evidence =evidence ,
    timestamp =datetime .now ().isoformat ()
    )

    return registrar .update_vrl (performance_evidence )

if __name__ =="__main__":
    logging .basicConfig (level =logging .INFO )

    registrar =MAMARegistrarService ()

    test_evidence =PerformanceEvidence (
    agent_id ="test_agent",
    task_id ="test_task_1",
    dimension ="reliability",
    score =0.85 ,
    evidence ={"test":"data"},
    timestamp =datetime .now ().isoformat ()
    )

    result =registrar .update_vrl (test_evidence )
    print (f"Update result: {result }")

    integrity =registrar .verify_integrity ()
    print (f"Integrity check: {integrity }")

    trust_score =registrar .get_trust_score ("test_agent")
    print (f"Trust score: {trust_score }")
