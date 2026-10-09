"""Check simulated connection timing without mutating state.
The minimum connection is 60 minutes. Date-specific schedules take precedence;
recorded actual timestamps remain authoritative.
"""
from datetime import datetime, timedelta
from collections.abc import Mapping

VERSION = 'connections_v1'
MIN_CONNECTION_MINUTES = 60

def scheduled(flight, date, field):
    dated = flight['dates'][date]
    value = dated.get(field, flight.get(field))
    if not isinstance(value, str):
        raise ValueError('Missing scheduled time')
    time, sep, days = value.partition('+')
    stamp = datetime.fromisoformat(date + 'T' + time)
    if stamp.tzinfo is not None:
        raise ValueError('Expected fixture-local EST clock')
    offset = int(days) if sep else 0
    if not 0 <= offset <= 7:
        raise ValueError('Invalid scheduled day offset')
    return stamp + timedelta(days=offset)

def timestamp(flight, date, kind):
    dated = flight['dates'][date]
    
    for prefix in ('actual', 'estimated'):
        value = dated.get(f'{prefix}_{kind}_time_est')
        if value:
            value = datetime.fromisoformat(value)
            if value.tzinfo is not None:
                raise ValueError('Expected fixture-local EST clock')
            return value
    return scheduled(flight, date, f'scheduled_{kind}_time_est')

def violations(state, segments, *, minimum_minutes=MIN_CONNECTION_MINUTES, active_only=False):
    if not isinstance(segments, list) or not segments:
        raise ValueError('Supply the complete nonempty itinerary')
    result = []
    for index, (first, next_leg) in enumerate(zip(segments, segments[1:])):
        a, b = state['flights'][first['flight_number']], state['flights'][next_leg['flight_number']]
        if active_only and all(f['dates'][s['date']]['status'] in {'landed','cancelled'} for f,s in ((a,first),(b,next_leg))):
            continue  
        arrival = timestamp(a, first['date'], 'arrival')
        departure = timestamp(b, next_leg['date'], 'departure')
        gap = (departure-arrival).total_seconds()/60
        if a['destination'] != b['origin'] or gap < minimum_minutes:
            result.append({'pair_index':index,'arrival_est':arrival.isoformat(),
                'next_departure_est':departure.isoformat(),'gap_minutes':gap,
                'airports_connect':a['destination']==b['origin']})
    return result

def check_airline_connections(state, tool_name, arguments):
    if tool_name not in {'book_reservation','update_reservation_flights'} or not isinstance(arguments,Mapping):
        return None
    if tool_name=='update_reservation_flights' and arguments.get('reservation_id') not in state.get('reservations',{}):
        return None  
    try:
        findings = violations(state,arguments.get('flights'),active_only=True)
    except (KeyError,TypeError,ValueError,OverflowError,AttributeError) as exc:
        return {'code':'AIRLINE_CONNECTION_FACTS_UNAVAILABLE',
            'message':f'Cannot establish connection timing: {exc}.', 'causal_inputs':['flights']}
    if findings:
        f=findings[0]
        return {'code':'AIRLINE_CONNECTION_NOT_FEASIBLE',
            'message':f"Adjacent flights must connect at the same airport with at least {MIN_CONNECTION_MINUTES} minutes between arrival and next departure. Invalid pair {f['pair_index']}: arrival {f['arrival_est']}, next departure {f['next_departure_est']}.",
            'causal_inputs':['flights']}
    return None

def apply_time_overrides(shared, overrides):
    """Detach only patched branches. Reset will deep-copy the resolved template."""
    from copy import deepcopy
    if not isinstance(overrides,dict):raise ValueError('Invalid flight time overrides')
    state={**shared,'flights':dict(shared['flights'])}
    allowed={'estimated_departure_time_est','estimated_arrival_time_est'}
    for number,dates in overrides.items():
        if number not in shared['flights'] or not isinstance(dates,dict):raise ValueError('Unknown overridden flight')
        state['flights'][number]=deepcopy(shared['flights'][number])
        for date,patch in dates.items():
            original=shared['flights'][number]['dates'][date]
            if not isinstance(patch,dict) or set(patch)!=allowed:raise ValueError('Time override must contain only paired estimates')
            if original['status'] not in {'available','on time','delayed'} or original.get('actual_departure_time_est'):raise ValueError('Cannot rewrite departed/cancelled flight timing')
            candidate=state['flights'][number];candidate['dates'][date].update(patch)
            old_dep=timestamp(shared['flights'][number],date,'departure');dep=timestamp(candidate,date,'departure')
            if dep<old_dep or dep-old_dep>timedelta(hours=24):raise ValueError('Delay must be forward and at most 24 hours')
            if timestamp(candidate,date,'arrival')-dep!=timestamp(shared['flights'][number],date,'arrival')-old_dep:raise ValueError('Flight duration changed')
    return state

def search_onestop(state, origin, destination, date):
    """Search connections using date-specific schedules and estimated timestamps."""
    import json
    from .airline_temporal import SIMULATION_NOW
    results=[]
    start=datetime.fromisoformat(date).date()
    def rendered(f,d):
        return {**{k:v for k,v in f.items() if k!='dates'},**f['dates'][d],'date':d}
    for first in state['flights'].values():
        if first['origin']!=origin or first['dates'].get(date,{}).get('status')!='available':continue
        if timestamp(first,date,'departure')<=SIMULATION_NOW:continue
        arrival=timestamp(first,date,'arrival')
        for second in state['flights'].values():
            if second['origin']!=first['destination'] or second['destination']!=destination or second['flight_number']==first['flight_number']:continue
            for d,instance in sorted(second['dates'].items()):
                service_date=datetime.fromisoformat(d).date()
                if not start<=service_date<=arrival.date()+timedelta(days=1) or instance.get('status')!='available':continue
                gap=timestamp(second,d,'departure')-arrival
                if timedelta(minutes=MIN_CONNECTION_MINUTES)<=gap<=timedelta(hours=24):
                    results.append([rendered(first,date),rendered(second,d)])
    return json.dumps(results)
