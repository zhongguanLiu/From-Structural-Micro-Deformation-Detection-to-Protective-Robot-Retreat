"""Recorded event lineage and shared-clock figures; no inferred planner timings."""
import numpy as np
import matplotlib.pyplot as plt


def event_chain(summary, states, metadata, evolution, evidence, maps, actions,
                plans, commands, truth, motion_time, exit_time, state_events=()):
    origin = states['MONITORING']
    end = states['COMPLETED']
    trigger = str(summary['return_trigger_algorithm_source_id'])
    series = sorted((r for r in evolution if r['gdu_id'] == trigger),
                    key=lambda r: float(r['message_time_s']))
    if not series:
        raise ValueError('No recorded evolution for the mission trigger GDU')
    ledger = []

    def add(name, source, stamp, receive=None, row=None, note=''):
        row = row or {}
        if source == 'events/mission_states.csv':
            matched = next((r for r in state_events if float(r['message_time_s']) == stamp), None)
            if matched and matched.get('receive_time_s') not in (None, ''):
                receive = float(matched['receive_time_s'])
        observed = stamp if stamp is not None else receive
        ledger.append(dict(event=name, source_file=source, source_time_s=stamp,
                           relative_to_monitoring_s=None if observed is None else observed-origin,
                           time_basis='source' if stamp is not None else ('receive_only' if receive is not None else 'unobserved'),
                           receive_time_s=receive,
                           snapshot_id=row.get('source_snapshot_id', ''),
                           frame_index=row.get('source_accepted_frame_index', row.get('accepted_frame_index', '')),
                           gdu_id=row.get('gdu_id', ''), plan_id=row.get('plan_id', ''),
                           goal_id=row.get('goal_id', ''), note=note))

    add('monitoring', 'events/mission_states.csv', origin)
    for state in ('RUNNING', 'HOLDING'):
        candidates = [r for r in truth if r['motion_state'] == state]
        if candidates:
            row = min(candidates, key=lambda r: float(r['truth_sim_time_s']))
            add('truth_'+state.lower(), 'truth/m01_m06_motion_truth.csv',
                float(row['truth_sim_time_s']), float(row['receive_time_s']),
                note='Recorded excitation phase, not semantic input to detection')
    strict = [r for r in evidence if r['gdu_id'] == trigger
              and r['detection_admission'] == '0' and r['detection_allowed'] == 'True']
    row = min(strict, key=lambda r: float(r['message_time_s'])) if strict else None
    add('strict_detection', 'deformation/detection_evidence.csv',
        float(row['message_time_s']) if row else None,
        float(row['receive_time_s']) if row else None, row,
        'Trigger GDU only; null means not observed, not zero latency')
    for field, name in (('active_now', 'first_active'), ('retained_for_avoidance', 'first_retained')):
        candidates = [r for r in series if r[field] == 'True']
        if candidates:
            row = candidates[0]
            add(name, 'evolution/gdu_evolution.csv', float(row['message_time_s']),
                float(row['receive_time_s']), row, 'Evolution activation, not strict detection admission')
    included = [r for r in maps if trigger in r['active_source_ids'].split('|')]
    if included:
        row = min(included, key=lambda r: float(r['message_time_s']))
        # Join exact upstream identities, not nearest time or spatial proximity.
        linked = any(r['source_snapshot_id'] == row['source_snapshot_id']
                     and r['source_accepted_frame_index'] == row['source_accepted_frame_index']
                     and r['retained_for_avoidance'] == 'True' for r in series)
        add('first_trigger_map', 'maps/avoidance_frames.csv', float(row['message_time_s']),
            float(row['receive_time_s']), row, 'Exact retained snapshot join: '+str(linked))
    add('persistence_decision', 'events/mission_states.csv', states['DEFORMATION_DETECTED'],
        note='Mission persistent-cause trigger; deformation recording cutoff')
    if 'return_trigger_retained_since_stamp_s' in summary:
        boundary = (summary['return_trigger_retained_since_stamp_s']
                    + summary['return_trigger_retained_duration_s'])
        candidates = [r for r in series if abs(float(r['message_time_s'])-boundary) < 1e-6
                      and r['retained_for_avoidance'] == 'True']
        add('persistence_source', 'mission_summary.json + evolution/gdu_evolution.csv', boundary,
            float(candidates[0]['receive_time_s']) if candidates else None,
            candidates[0] if candidates else None,
            'Reported timer endpoint; exact retained row found: '+str(bool(candidates)))
    add('map_freeze', 'navigation/maps/decision_map.npz', metadata['freeze_time_s'],
        note='Freeze time, not overlay measurement time')
    add('return_state', 'events/mission_states.csv', states['RETURNING_HOME'])
    goals = [r for r in actions if r['event'] == 'goal'
             and states['RETURNING_HOME'] <= float(r['ros_time_s']) <= end]
    goal = min(goals, key=lambda r: float(r['ros_time_s'])) if goals else None
    goal_time = float(goal['message_time_s']) if goal else states['RETURNING_HOME']
    if goal:
        add('return_goal', 'navigation/action_events.csv', goal_time,
            float(goal['ros_time_s']), goal)
    # Preserve every publication, including empty paths. A Path has no goal ID.
    publications = []
    grouped = {}
    for r in plans:
        if goal_time <= float(r['message_time_s']) <= end and float(r['ros_time_s']) <= end:
            grouped.setdefault(r['plan_id'], []).append(r)
    for plan_id, group in grouped.items():
        r = group[0]
        valid = sorted((p for p in group if int(p['point_index']) >= 0),
                       key=lambda p: int(p['point_index']))
        if valid and ([int(p['point_index']) for p in valid] != list(range(int(r['point_count'])))
                      or r['frame_id'] != metadata['frame_id']):
            raise ValueError('Incomplete or wrong-frame recorded path: '+plan_id)
        xy = np.array([[float(p['x_m']), float(p['y_m'])] for p in valid])
        publications.append(dict(plan_id=plan_id, source_time_s=float(r['message_time_s']),
                                 receive_time_s=float(r['ros_time_s']), point_count=len(valid),
                                 length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) else None))
    publications.sort(key=lambda r: r['receive_time_s'])
    valid = [r for r in publications if r['point_count']]
    if valid:
        r = valid[0]
        add('first_return_path', 'navigation/global_plans.csv', r['source_time_s'],
            r['receive_time_s'], r, 'First nonempty publication after goal timestamp; no exact action/solver binding')
    return_controls = [r for r in commands if states['RETURNING_HOME'] <= float(r['ros_time_s']) <= end
              and (np.hypot(float(r['vx_mps']), float(r['vy_mps'])) > .01
                   or abs(float(r['wz_rad_s'])) > .01)]
    if return_controls:
        r = min(return_controls, key=lambda r: float(r['ros_time_s']))
        add('first_return_control', 'trajectories/cmd_vel.csv', None, float(r['ros_time_s']),
            note='First nonzero return-phase control, including pre-goal rotation; unstamped Twist')
    moving = [r for r in return_controls if float(r['ros_time_s']) >= goal_time]
    if moving:
        r = min(moving, key=lambda r: float(r['ros_time_s']))
        add('first_motion_command', 'trajectories/cmd_vel.csv', None, float(r['ros_time_s']),
            note='Unstamped Twist; receive time only; >0.01 m/s OR rad/s')
    add('measured_motion', 'trajectories/navigation_base_odometry.csv', motion_time,
        note='First sampled translation >=0.05 m from initial return pose')
    add('first_sustained_exit', 'trajectories/navigation_base_odometry.csv + frozen map', exit_time,
        note='Retrospective interval start, confirmed after >=1 s; later re-entry is possible')
    add('arrival', 'events/mission_states.csv', end)
    by_name = {r['event']: r for r in ledger}
    intervals = []
    for start, stop, basis, meaning in (
            ('truth_running', 'first_active', 'source_time_s', 'Excitation phase to trigger-GDU active evidence; not object-specific accuracy'),
            ('first_active', 'first_retained', 'source_time_s', 'Temporal birth confirmation'),
            ('first_retained', 'first_trigger_map', 'source_time_s', 'Shared upstream data timestamp; NOT map compute time'),
            ('first_retained', 'persistence_source', 'source_time_s', 'Recorded continuous retention duration'),
            ('persistence_source', 'persistence_decision', 'source_time_s', 'Source evidence to mission decision; scheduling/processing combined'),
            ('persistence_decision', 'return_state', 'source_time_s', 'Mission preparation gap (capture/export policy and motion stop); not planner solve time'),
            ('return_state', 'return_goal', 'source_time_s', 'Return setup including checked heading alignment; not planner solve time'),
            ('return_goal', 'first_return_path', 'source_time_s', 'Goal to first subsequent path publication; no exact solver binding'),
            ('return_goal', 'first_motion_command', 'receive_time_s', 'Recorder-observed goal to first nonzero command'),
            ('measured_motion', 'first_sustained_exit', 'source_time_s', 'Motion to first sustained exit, not permanent exit'),
            ('return_state', 'arrival', 'source_time_s', 'Complete return')):
        a, b = by_name.get(start, {}).get(basis), by_name.get(stop, {}).get(basis)
        intervals.append(dict(start_event=start, end_event=stop, clock=basis,
                              duration_sim_s=None if a is None or b is None else b-a,
                              interpretation=meaning))
    ledger.sort(key=lambda r: r['relative_to_monitoring_s'] if r['relative_to_monitoring_s'] is not None else float('inf'))
    return ledger, intervals, publications, series


def path_following(plans, publications, times, poses, origin):
    """Distance to the last path received by each pose time, not hindsight best path."""
    geometry = {}
    for r in plans:
        if int(r['point_index']) >= 0:
            geometry.setdefault(r['plan_id'], []).append(r)
    geometry = {key: np.array([[float(r['x_m']), float(r['y_m'])] for r in
                              sorted(group, key=lambda r: int(r['point_index']))])
                for key, group in geometry.items()}
    stamps = np.array([r['receive_time_s'] for r in publications])
    result = []
    for stamp, pose in zip(times, poses):
        index = np.searchsorted(stamps, stamp, side='right')-1
        publication = publications[index] if index >= 0 else None
        xy = geometry.get(publication['plan_id']) if publication else None
        distance = None
        if xy is not None and publication['point_count']:
            if len(xy) == 1:
                distance = float(np.linalg.norm(pose[:2]-xy[0]))
            else:
                segment = np.diff(xy, axis=0)
                length2 = (segment**2).sum(axis=1)
                fraction = np.divide(((pose[:2]-xy[:-1])*segment).sum(axis=1), length2,
                                     out=np.zeros(len(segment)), where=length2 > 0)
                nearest = xy[:-1]+np.clip(fraction, 0, 1)[:, None]*segment
                distance = float(np.min(np.linalg.norm(nearest-pose[:2], axis=1)))
        result.append(dict(source_time_s=float(stamp), relative_to_monitoring_s=float(stamp-origin),
                           plan_id=publication['plan_id'] if publication else '',
                           path_receive_age_s=float(stamp-publication['receive_time_s']) if publication else None,
                           distance_to_last_published_path_m=distance))
    return result


# Shared with the spatial process map: blue outbound/evidence, green return,
# burgundy deformation cost. Colour never encodes a calibrated collapse probability.
OUTBOUND_COLOR = '#2166ac'
RETURN_COLOR = '#007f69'
DEFORMATION_COLOR = '#ad3551'


def plot_chain(out, ledger, series, times, costs, states):
    """Single-column evidence / executed-exposure pair, with one trigger origin."""
    event = {r['event']: r for r in ledger}
    monitor, decision, end = (states[k] for k in ('MONITORING', 'DEFORMATION_DETECTED', 'COMPLETED'))
    observed = sorted((r for r in series if monitor <= float(r['message_time_s']) <= decision),
                      key=lambda r: float(r['message_time_s']))
    et = np.array([float(r['message_time_s']) for r in observed])
    ed = np.array([1000*float(r['displacement_rms_m']) for r in observed])
    times, costs = np.asarray(times, dtype=float), np.asarray(costs, dtype=float)
    if (len(et) < 2 or not np.isfinite(ed).all() or np.any(ed < 0)
            or np.any(np.diff(et) <= 0) or times.ndim != 1 or costs.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid evidence or exposure series')
    keep = (times >= states['RETURNING_HOME']) & (times <= end)
    times, costs = times[keep], costs[keep]
    if (len(times) < 2 or not np.isfinite(costs).any()
            or np.any(np.isinf(costs)) or np.any((costs < 0) | (costs > 99))):
        raise ValueError('Invalid recorded return costs')
    # Unknown evaluations and gaps remain gaps, never zeros or invented plateaus.
    gaps = np.flatnonzero(np.diff(times) > .5) + 1
    rt, rc = np.insert(times-decision, gaps, np.nan), np.insert(costs, gaps, np.nan)
    first = event.get('first_retained', {}).get('source_time_s')
    last = event.get('persistence_source', {}).get('source_time_s')
    if first is not None and last is not None and not monitor <= first <= last <= decision:
        raise ValueError('Persistence interval outside recorded observation window')

    with plt.rc_context({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'pdf.fonttype': 42, 'savefig.dpi': 450}):
        fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.55), layout='constrained')
        a.plot(et-decision, ed, color=OUTBOUND_COLOR, lw=1.35)
        a.fill_between(et-decision, ed, color=OUTBOUND_COLOR, alpha=.09)
        if first is not None and last is not None:
            a.plot([first-decision, last-decision], [.86, .86], '|-', color=DEFORMATION_COLOR,
                   ms=5, lw=.9, transform=a.get_xaxis_transform())
            a.text((first+last)/2-decision, .91, '{:.2f} s'.format(last-first),
                   ha='center', transform=a.get_xaxis_transform(), fontsize=7, color=DEFORMATION_COLOR)
        a.set(xlim=(monitor-decision, .5), ylim=(0, max(ed.max()*1.3, 1)),
              ylabel='Estimated displacement (mm)')
        a.set_title('(a) Deformation evidence', loc='left', fontsize=8)
        b.plot(rt, rc, color=DEFORMATION_COLOR, lw=1.35)
        b.fill_between(rt, rc, color=DEFORMATION_COLOR, alpha=.1)
        b.set(xlim=(-.5, end-decision+.5), ylim=(-2, 102), yticks=[0, 25, 50, 75, 99],
              ylabel='Footprint deformation cost')
        b.set_title('(b) Exposure during return', loc='left', fontsize=8)
        for ax in (a, b):
            ax.axvline(0, color=DEFORMATION_COLOR, ls='--', lw=.8, alpha=.8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#d9dde1', lw=.45, alpha=.7)
            ax.tick_params(direction='out', length=2.5)
            ax.set_axisbelow(True)
        fig.supxlabel('Time relative to return trigger (s)', fontsize=7.5)
        for ext in ('pdf', 'png'):
            fig.savefig(out / ('fig_evacuation_response.'+ext))
        plt.close(fig)
