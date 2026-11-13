"""Prepare Midtown Citi Bike model inputs from the project root.

Run from the project root with:
    python preparation/data_preparation.py

Dependencies: numpy, pandas, matplotlib, requests, shapely, pyproj,
scikit-learn and threadpoolctl.
"""

import matplotlib


def main():
    matplotlib.use("Agg")
    # ## Step 1: Research scope
    #
    # | Item | Definition |
    # |---|---|
    # | Study area | Citi Bike stations on Manhattan's main island; exclude other boroughs and separate islands |
    # | Regions | Lower: south of 14th Street; Midtown: 14th–59th Street; Upper: north of 59th Street |
    # | Boundaries | Street centerlines and the island polygon; a shared boundary belongs to one region only |
    # | Station scope | All historical station IDs in the study area, without a sample-size cap; audit renumbering, openings and closures |
    # | Historical data | All CSV parts of the official July, August and September 2026 trip ZIPs |
    # | Analysis dates | July 2–September 29, 2026; trim the endpoint days to reduce cross-month omissions |
    # | Weekdays | Monday–Friday, including public holidays |
    # | Morning peak | 06:00 <= local New York time < 10:00 |
    # | Pickups P | Completed trip starts, counted by starting station and time |
    # | Returns R | Completed trip ends, counted by ending station and time; include arrivals from outside the area |
    # | Demand units | Events per station per weekday morning peak |
    # | Static demand | One fixed P and R estimate per station over the selected forecast dates |
    #
    # The 14th/59th Street division is an operational research definition, not the only official neighborhood classification. Later steps fix the inventory case at midnight on October 28 and add optimization.

    # ### Data sources and roles
    #
    # - Official trip records support the regional comparison and pickup/return prediction. Timely inventory charts on ctbk.dev do not imply that monthly trip records have no publication delay.
    # - Historical GBFS records from ctbk.dev provide the midnight inventory case, timestamps and status checks.
    # - Station metadata and historical endpoint coordinates support identity, location and capacity checks. A current roster alone cannot identify every historical station.
    # - Main-island and street boundaries define the geographic scope.
    #
    # Sources: [Citi Bike system data](https://citibikenyc.com/system-data), [ctbk.dev](https://ctbk.dev/), [ctbk data pipeline](https://github.com/hudcostreets/ctbk.dev/blob/main/docs/pipeline.md).

    # ### Selecting a region from data
    # Compare historical observations before fitting prediction models:
    #
    # 1. Station counts, dates and coverage.
    # 2. Total pickup/return activity and activity per station.
    # 3. Station-level net flow P−R, its direction and spatial distribution.
    # 4. Both regional net flow and the sum of station-level absolute net flows, so opposing flows do not cancel invisibly.
    #
    # These statistics describe demand patterns and research relevance. They do not prove optimal rebalancing benefits; the largest trip total alone is insufficient.

    # ## Step 2: Read data, audit quality and map the three regions
    # Check whether all files were read, which station IDs appear, and whether the regions cover the main island without overlap. Historical IDs are statistical units, not automatically distinct physical stations. Audit large location changes and cross-boundary coordinates separately.
    #
    # Boundary sources: NYC DCP [Borough Boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne) and NYC [Centerline](https://data.cityofnewyork.us/resource/inkn-q76z.geojson). Use the largest connected Manhattan land polygon; exclude Roosevelt Island, Governors Island and Marble Hill.
    #
    # Interpolate the E/W 14 ST, E/W 59 ST and CENTRAL PARK S centerlines from west to east and extend their ends to the shoreline. Where branches share an x-coordinate, use the median y-coordinate. This reproducible approximation is not a claim of exact street-side classification. List stations within 30 m of either dividing line for review; shared boundaries belong to the northern region.

    from pathlib import Path
    import json, zipfile, urllib.request, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import requests
    from shapely.geometry import shape, Point, LineString
    from shapely.ops import transform
    from pyproj import Transformer
    ROOT=Path.cwd()
    RAW=ROOT/'data/raw'
    GEO=ROOT/'data/geography'
    OUT=ROOT/'outputs_stage2'
    for folder in [RAW,GEO,OUT]:
        folder.mkdir(parents=True,exist_ok=True)
    MONTHS=['202607','202608','202609']
    START,END=pd.Timestamp('2026-07-02'),pd.Timestamp('2026-09-29')
    REGION_ORDER=['Lower Manhattan','Midtown','Upper Manhattan']

    # Download geography once; keep raw files in data/geography.
    geo_sources={
     'manhattan_borough.geojson':('https://data.cityofnewyork.us/resource/gthc-hcne.geojson',{'$where':"boroname='Manhattan'"}),
     'street_14.geojson':('https://data.cityofnewyork.us/resource/inkn-q76z.geojson',{'$where':"boroughcode='1' AND (full_street_name='W  14 ST' OR full_street_name='E  14 ST')",'$limit':5000}),
     'street_59.geojson':('https://data.cityofnewyork.us/resource/inkn-q76z.geojson',{'$where':"boroughcode='1' AND (full_street_name='W  59 ST' OR full_street_name='E  59 ST' OR full_street_name='CENTRAL PARK S')",'$limit':5000})}
    for filename,(url,params) in geo_sources.items():
        path=GEO/filename
        if not path.exists():
            response=requests.get(url,params=params,timeout=60)
            response.raise_for_status()
            path.write_text(response.text,encoding='utf-8')
        if not json.loads(path.read_text())['features']:
            raise ValueError('No geographic features returned: '+filename)
    project=Transformer.from_crs(4326,32618,always_xy=True).transform
    borough=shape(json.loads((GEO/'manhattan_borough.geojson').read_text())['features'][0]['geometry'])
    land=transform(project,borough)
    island=max(land.geoms,key=lambda g:g.area) if land.geom_type=='MultiPolygon' else land
    street_lines={}
    vertices={}
    for number in [14,59]:
        lines=[]
        for f in json.loads((GEO/f'street_{number}.geojson').read_text())['features']:
            g=transform(project,shape(f['geometry']))
            lines.extend(list(g.geoms) if g.geom_type=='MultiLineString' else [g])
        points=np.vstack([np.asarray(g.coords) for g in lines])
        v=pd.DataFrame(points,columns=['x','y']).groupby('x',as_index=False).y.median().sort_values('x')
        vertices[number]=v
        street_lines[number]=lines

    def boundary_y(x,number):
        v=vertices[number]
        xx=v.x.to_numpy()
        yy=v.y.to_numpy()
        x=np.asarray(x)
        out=np.interp(x,xx,yy)
        # Extend endpoints using an approximately 300 m direction to reduce short-segment noise.
        left=min(np.searchsorted(xx,xx[0]+300),len(xx)-1)
        right=max(np.searchsorted(xx,xx[-1]-300),0)
        out=np.where(x<xx[0],yy[0]+(x-xx[0])*(yy[left]-yy[0])/(xx[left]-xx[0]),out)
        out=np.where(x>xx[-1],yy[-1]+(x-xx[-1])*(yy[-1]-yy[right])/(xx[-1]-xx[right]),out)
        return out
    xs=np.linspace(island.bounds[0]-100,island.bounds[2]+100,1000)
    assert (boundary_y(xs,59)>boundary_y(xs,14)).all(), 'Boundary lines intersect; review the geometry.'
    boundaries={k:LineString(np.column_stack([xs,boundary_y(xs,k)])) for k in [14,59]}
    print('Main-island area (square km):',round(island.area/1e6,2))

    # ### 2.1 Read every month and CSV part
    # Deduplicate by ride_id within and across months. Missing IDs cannot be reliably deduplicated and are excluded separately. Validate the start and end independently, retaining a usable endpoint even when the other is missing.
    #
    # Include cross-region trips. Represent station locations by event-weighted mean coordinates over valid endpoints. Review IDs with coordinate spans above 100 m or coordinates crossing regions; these checks cannot automatically distinguish relocation from renumbering.
    #
    # Daily system activity does not establish daily operation at every station. Retain first/last observations and active-day counts; do not silently treat unobserved station-days as zero-demand operating days.

    # Bind cache validity to raw file sizes and analysis dates; avoid rereading every ZIP.
    cache_files=['source_quality.csv','all_station_geography_audit.csv','station_daily_activity.csv','peak_counts_all_stations.csv']
    signature={'version':1,'start':str(START.date()),'end':str(END.date()),'raw_sizes':{m:(RAW/f'{m}-citibike-tripdata.zip').stat().st_size if (RAW/f'{m}-citibike-tripdata.zip').exists() else None for m in MONTHS}}
    manifest=OUT/'cache_manifest.json'
    cache_ok=manifest.exists() and all((OUT/f).exists() for f in cache_files)
    if cache_ok:
        old=json.loads(manifest.read_text())
        # Bundled caches support missing local raw ZIPs; existing raw ZIP sizes must match.
        cache_ok=old['version']==signature['version'] and old['start']==signature['start'] and old['end']==signature['end'] and all(size is None or size==old['raw_sizes'][m] for m,size in signature['raw_sizes'].items())
    if cache_ok:
        quality=pd.read_csv(OUT/'source_quality.csv')
        stations=pd.read_csv(OUT/'all_station_geography_audit.csv',dtype={'station_id':str},parse_dates=['first_date','last_date'])
        all_days=pd.read_csv(OUT/'station_daily_activity.csv',dtype={'station_id':str},parse_dates=['date'])
        peak_counts=pd.read_csv(OUT/'peak_counts_all_stations.csv',dtype={'station_id':str},parse_dates=['date'])
        print('Loaded cached real-data processing results; raw ZIPs were not reread.')
        print(quality)
    else:
        station_parts=[]
        endpoint_parts=[]
        audit=[]
        seen_previous=set()
        all_day_parts=[]
        for month in MONTHS:
            path=RAW/f'{month}-citibike-tripdata.zip'
            url=f'https://tripdata.s3.amazonaws.com/{path.name}'
            if not path.exists():
                print('Downloading',month,flush=True)
                tmp=path.with_suffix('.part')
                urllib.request.urlretrieve(url,tmp)
                if not zipfile.is_zipfile(tmp):
                    raise ValueError('Downloaded file is not a ZIP.')
                tmp.replace(path)
            total=missing_id=duplicates=bad_time=missing_sid=bad_coords=0
            seen_month=set()
            with zipfile.ZipFile(path) as z:
                members=[s for s in z.namelist() if s.endswith('.csv') and not s.startswith('__MACOSX/')]
                if not members:
                    raise ValueError('ZIP contains no CSV files.')
                for member in members:
                    with z.open(member) as stream:
                        columns=['ride_id','started_at','ended_at','start_station_id','end_station_id','start_station_name','end_station_name','start_lat','start_lng','end_lat','end_lng']
                        for chunk in pd.read_csv(stream,usecols=columns,dtype=str,chunksize=200000):
                            total+=len(chunk)
                            missing_id+=int(chunk.ride_id.isna().sum())
                            chunk=chunk.dropna(subset=['ride_id'])
                            dup=chunk.ride_id.map(seen_month.__contains__)|chunk.ride_id.map(seen_previous.__contains__)|chunk.ride_id.duplicated()
                            duplicates+=int(dup.sum())
                            seen_month.update(chunk.ride_id)
                            chunk=chunk.loc[~dup]
                            for prefix,tc,flow in [('start','started_at','P'),('end','ended_at','R')]:
                                t=pd.to_datetime(chunk[tc],errors='coerce',format='mixed')
                                sid=chunk[prefix+'_station_id'].str.strip().replace('',np.nan)
                                lat=pd.to_numeric(chunk[prefix+'_lat'],errors='coerce')
                                lon=pd.to_numeric(chunk[prefix+'_lng'],errors='coerce')
                                bad_time+=int(t.isna().sum())
                                missing_sid+=int(sid.isna().sum())
                                valid_coord=lat.between(40,42)&lon.between(-75,-72)
                                bad_coords+=int((~valid_coord).sum())
                                period=t.dt.normalize().between(START,END)
                                q=pd.DataFrame({'date':t.dt.normalize(),'station_id':sid,'name':chunk[prefix+'_station_name'],'lat':lat,'lon':lon})
                                valid=period & sid.notna() & valid_coord & t.notna()
                                q=q.loc[valid].copy()
                                # Discover stations using all-day activity, including stations with no morning trips.
                                summary=q.groupby('station_id').agg(n=('lat','size'),lat_sum=('lat','sum'),lon_sum=('lon','sum'),lat_min=('lat','min'),lat_max=('lat','max'),lon_min=('lon','min'),lon_max=('lon','max'),first_date=('date','min'),last_date=('date','max'),name=('name','first')).reset_index()
                                station_parts.append(summary)
                                allday=q.groupby(['date','station_id']).size().rename('events').reset_index()
                                all_day_parts.append(allday)
                                peak=q[t.loc[q.index].dt.hour.between(6,9)].groupby(['date','station_id']).size().rename('count').reset_index()
                                peak['flow']=flow
                                endpoint_parts.append(peak)
            seen_previous.update(seen_month)
            audit.append({'month':month,'csv_parts':len(members),'raw_rows':total,'missing_ride_id':missing_id,'duplicate_rides':duplicates,'bad_endpoint_time':bad_time,'missing_endpoint_station_id':missing_sid,'invalid_endpoint_coordinates':bad_coords,'zip_bytes':path.stat().st_size})
            print(month,'Read all CSV parts:',total,'rows',flush=True)
        # Missingness counts are overlapping endpoint counts, not additive excluded-trip totals.
        quality=pd.DataFrame(audit)
        print(quality)
        s=pd.concat(station_parts)
        stations=s.groupby('station_id').agg(n=('n','sum'),lat_sum=('lat_sum','sum'),lon_sum=('lon_sum','sum'),lat_min=('lat_min','min'),lat_max=('lat_max','max'),lon_min=('lon_min','min'),lon_max=('lon_max','max'),first_date=('first_date','min'),last_date=('last_date','max'),name=('name','first')).reset_index()
        stations['lat']=stations.lat_sum/stations.n
        stations['lon']=stations.lon_sum/stations.n
        stations['coordinate_span_m']=np.hypot((stations.lat_max-stations.lat_min)*111320,(stations.lon_max-stations.lon_min)*111320*np.cos(np.deg2rad(stations.lat)))
        all_days=pd.concat(all_day_parts).groupby(['date','station_id']).events.sum().reset_index()
        active=all_days.groupby('station_id').date.nunique()
        stations['active_days']=stations.station_id.map(active)
        peak_counts=pd.concat(endpoint_parts).groupby(['date','station_id','flow'])['count'].sum().reset_index()
        quality.to_csv(OUT/'source_quality.csv',index=False)
        peak_counts.to_csv(OUT/'peak_counts_all_stations.csv',index=False)
        all_days.to_csv(OUT/'station_daily_activity.csv',index=False)
        # Release raw IDs and temporary aggregates; subsequent stages use smaller tables.
        del seen_previous,seen_month,station_parts,endpoint_parts,all_day_parts,s

    # ### 2.2 Main-island membership and boundary review
    # Keep every historical ID classified inside the main island. Do not snap nearby offshore points into the polygon; list them separately. Exclusions do not establish complete coverage.

    x,y=project(stations.lon.to_numpy(),stations.lat.to_numpy())
    stations['x']=x
    stations['y']=y
    points=[Point(a,b) for a,b in zip(x,y)]
    stations['on_main_island']=[island.covers(p) for p in points]
    stations['region']=np.where(y<boundary_y(x,14),'Lower Manhattan',np.where(y<boundary_y(x,59),'Midtown','Upper Manhattan'))
    stations.loc[~stations.on_main_island,'region']='Outside main island'
    main_stations=stations[stations.on_main_island].copy()
    assert main_stations.station_id.is_unique
    assert main_stations.region.isin(REGION_ORDER).all()
    coverage=all_days[all_days.station_id.isin(main_stations.station_id)].groupby('date').events.sum().reindex(pd.date_range(START,END),fill_value=0)
    print('Dates:',len(coverage),'Weekdays:',int((coverage.index.dayofweek<5).sum()),'Main-island dates without valid events:',int(coverage.eq(0).sum()))
    if coverage.eq(0).any():
        raise ValueError('A date has no main-island events; stop and investigate.')
    stations.to_csv(OUT/'all_station_geography_audit.csv',index=False)

    # Flag numeric-format aliases; do not merge without checking physical identity.
    from decimal import Decimal, InvalidOperation
    import re
    def decimal_key(value):
        if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',value):
            return value
        try:
            return format(Decimal(value).normalize(),'f')
        except InvalidOperation:
            return value
    main_stations['numeric_id_key']=main_stations.station_id.map(decimal_key)
    alias_review=main_stations[main_stations.numeric_id_key.duplicated(keep=False)].sort_values('numeric_id_key')
    # Create the cache signature only after processing completes.
    if not cache_ok:
        manifest.write_text(json.dumps(signature,indent=2),encoding='utf-8')

    # Merge verified numeric-format aliases with matching names/coordinates; preserve suffixed IDs.
    main_stations['canonical_station_id']=main_stations.station_id
    alias_decisions=[]
    for key,g in alias_review.groupby('numeric_id_key'):
        same_name=g.name.fillna('').str.strip().nunique()==1
        max_distance=np.hypot(g.x.max()-g.x.min(),g.y.max()-g.y.min())
        ok=same_name and max_distance<=5
        canonical=sorted(g.station_id,key=lambda v:(-len(v),v))[0]
        if ok:
            main_stations.loc[g.index,'canonical_station_id']=canonical
        for sid in g.station_id:
            alias_decisions.append({'station_id':sid,'canonical_station_id':canonical if ok else sid,'merged_format_alias':ok,'coordinate_distance_m':max_distance})
    pd.DataFrame(alias_decisions).to_csv(OUT/'alias_resolution.csv',index=False)
    canonical_map=main_stations.set_index('station_id').canonical_station_id
    canonical_stations=main_stations.drop_duplicates('canonical_station_id').copy()
    canonical_stations['station_id']=canonical_stations.canonical_station_id
    main_stations.to_csv(OUT/'manhattan_station_regions.csv',index=False)
    canonical_stations.to_csv(OUT/'manhattan_canonical_stations.csv',index=False)
    for frame,value,filename in [(peak_counts,'count','manhattan_daily_peak_counts.csv'),(all_days,'events','manhattan_daily_activity.csv')]:
        d=frame[frame.station_id.isin(canonical_map.index)].copy()
        d['station_id']=d.station_id.map(canonical_map)
        keys=['date','station_id']+(['flow'] if 'flow' in d else [])
        d=d.groupby(keys,as_index=False)[value].sum()
        d.to_csv(OUT/filename,index=False)
    print('Only verified format aliases are merged; renumbering and relocation are not fully resolved.')

    # ## Step 3: Compare the three regions
    # Use only the 43 weekdays from July 2–August 31 for regional selection. Reserve September for validation and testing. Geographic identity tables support retrospective research preparation.

    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    BASE3=Path.cwd()
    INPUT3=BASE3/'outputs_stage2'
    OUT3=BASE3/'outputs_stage3'
    OUT3.mkdir(exist_ok=True)
    ORDER3=['Lower Manhattan','Midtown','Upper Manhattan']
    COLORS3={'Lower Manhattan':'#2b8cbe','Midtown':'#e69f00','Upper Manhattan':'#009e73'}
    SELECT_START3=pd.Timestamp('2026-07-02')
    SELECT_END3=pd.Timestamp('2026-08-31')
    DAYS3=pd.bdate_range(SELECT_START3,SELECT_END3)
    station3=pd.read_csv(INPUT3/'manhattan_canonical_stations.csv',dtype={'station_id':str,'canonical_station_id':str})
    identity3=pd.read_csv(INPUT3/'manhattan_station_regions.csv',dtype={'station_id':str,'canonical_station_id':str})
    peak3=pd.read_csv(INPUT3/'manhattan_daily_peak_counts.csv',dtype={'station_id':str},parse_dates=['date'])
    activity3=pd.read_csv(INPUT3/'manhattan_daily_activity.csv',dtype={'station_id':str},parse_dates=['date'])
    assert len(DAYS3)==43 and station3.station_id.is_unique
    period3=activity3[activity3.date.isin(DAYS3)]
    observed_days3=period3.groupby('station_id').date.nunique()
    selection_stations3=station3[station3.station_id.isin(observed_days3.index)].copy()
    selection_stations3['days_with_any_event']=selection_stations3.station_id.map(observed_days3)
    new_later3=station3[~station3.station_id.isin(observed_days3.index)]
    print('Selection weekdays:',len(DAYS3),'Analysis IDs observed in the selection period:',len(selection_stations3),'IDs first appearing later:',len(new_later3))
    print(new_later3[['station_id','name','region']])

    # ### 3.2 Comparison metrics
    # Compute each station's mean pickups P and returns R over the 43 selection weekdays.
    #
    # | Metric | Calculation | Interpretation and limitation |
    # |---|---|---|
    # | Total activity A | Σ(P+R) | Endpoint events per morning; an internal trip may contribute two endpoints |
    # | Activity per station | A / analysis ID count | Separates scale from typical station activity; not demand per unit area |
    # | Absolute net flow B | Σ\|P−R\| | Station directionality, not required truck transfers |
    # | Absolute net flow per station | B / analysis ID count | Average directional imbalance |
    # | Regional net flow | Σ(P−R) | Positive means net departures; opposing station flows may cancel |
    # | Opposing-flow scale | min(Σmax(P−R,0), Σmax(R−P,0)) | Whether both flow directions have substantial scale; not proven transferable inventory |

    selection_peak3=peak3[peak3.date.isin(DAYS3)&peak3.station_id.isin(selection_stations3.station_id)].copy()
    index3=pd.MultiIndex.from_product([DAYS3,selection_stations3.station_id,['P','R']],names=['date','station_id','flow'])
    panel3=selection_peak3.set_index(['date','station_id','flow'])['count'].reindex(index3,fill_value=0).unstack('flow').reset_index()
    assert not panel3[['P','R']].isna().any().any()
    means3=panel3.groupby('station_id')[['P','R']].mean().reset_index()
    station_stats3=selection_stations3[['station_id','name','region','lat','lon','x','y','days_with_any_event']].merge(means3,on='station_id',validate='one_to_one')
    station_stats3['net']=station_stats3.P-station_stats3.R
    station_stats3['activity']=station_stats3.P+station_stats3.R
    station_stats3['abs_net']=station_stats3.net.abs()
    station_stats3['net_pickup']=station_stats3.net.clip(lower=0)
    station_stats3['net_return']=(-station_stats3.net).clip(lower=0)

    def summarize_regions3(s):
        g=s.groupby('region').agg(station_ids=('station_id','size'),pickups=('P','sum'),returns=('R','sum'),activity=('activity','sum'),sum_abs_station_net=('abs_net','sum'),region_net=('net','sum'),net_pickup_sum=('net_pickup','sum'),net_return_sum=('net_return','sum')).reindex(ORDER3)
        g['activity_per_station']=g.activity/g.station_ids
        g['abs_net_per_station']=g.sum_abs_station_net/g.station_ids
        g['imbalance_share']=g.sum_abs_station_net/g.activity
        g['opposing_flow_scale']=np.minimum(g.net_pickup_sum,g.net_return_sum)
        return g
    summary3=summarize_regions3(station_stats3)
    # Use +/-1 event per morning as a descriptive near-zero band, not a station filter.
    station_stats3['direction']=np.where(station_stats3.net>1,'net pickup',np.where(station_stats3.net< -1,'net return','near balanced'))
    directions3=pd.crosstab(station_stats3.region,station_stats3.direction).reindex(ORDER3).fillna(0)
    daily3=panel3.merge(station_stats3[['station_id','region']],on='station_id',validate='many_to_one')
    daily3['daily_abs_net']=(daily3.P-daily3.R).abs()
    summary3['mean_daily_sum_abs_net']=daily3.groupby(['region','date']).daily_abs_net.sum().groupby('region').mean()
    summary3.to_csv(OUT3/'region_comparison_selection_period.csv')
    print(summary3.round(1))
    print(directions3)
    assert np.isclose(summary3.pickups.sum(),selection_peak3.loc[selection_peak3.flow=='P','count'].sum()/len(DAYS3))
    assert np.isclose(summary3.returns.sum(),selection_peak3.loc[selection_peak3.flow=='R','count'].sum()/len(DAYS3))

    # Generate the recommendation from actual results, without hard-coding a preferred region.
    leader_activity3=summary3.activity.idxmax()
    leader_perstation3=summary3.activity_per_station.idxmax()
    leader_imbalance3=summary3.abs_net_per_station.idxmax()
    strong3=len({leader_activity3,leader_perstation3,leader_imbalance3})==1
    recommendation3=leader_activity3 if strong3 else 'Trade-offs require further review; no automatic selection.'
    print('Suggested priority region (subject to team agreement):',recommendation3)
    print('Largest opposing-flow scale:',summary3.opposing_flow_scale.idxmax())
    print('This does not establish actual rebalancing gains; later stages determine the case and inputs.')
    (OUT3/'recommendation.json').write_text(json.dumps({'selection_start':str(SELECT_START3.date()),'selection_end':str(SELECT_END3.date()),'weekday_count':len(DAYS3),'recommended_candidate':recommendation3,'confirmed':False,'optimization_run':False,'prediction_model_trained':False},ensure_ascii=False,indent=2),encoding='utf-8')

    # ### Regional decision: Midtown
    #
    # Use Midtown for the case study based on the baseline regional comparison.

    # ## Step 4: Predict pickups and returns across Midtown
    # Retain all **190 historical analysis IDs** classified within the main island between 14th and 59th Street. Historical IDs do not establish which physical stations operate on the later case date.
    #
    # Output one fixed P_i and R_i per station, measured in events/station/weekday morning peak, 06:00–10:00. Predict specific dates first, then average over five weekdays. This stage alone does not produce initial inventories, target stocks or routes.

    from pathlib import Path
    import json, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import PoissonRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    import sklearn
    ROOT4=Path.cwd()
    IN4=ROOT4/'outputs_stage2'
    OUT4=ROOT4/'outputs_stage4'
    OUT4.mkdir(exist_ok=True)
    TRAIN_END4=pd.Timestamp('2026-08-31')
    VAL_END4=pd.Timestamp('2026-09-15')
    HISTORY_END4=pd.Timestamp('2026-09-29')
    HISTORY_START4=pd.Timestamp('2026-07-02')
    FORECAST_START4=pd.Timestamp('2026-10-29')
    FUTURE_DATES4=pd.bdate_range(FORECAST_START4,periods=5)
    ALPHA4=0.001  # Predefined weak regularization; no validation/test hyperparameter tuning.
    MIN_HISTORY_DAYS4=20  # Manual-review flag only; do not exclude stations.
    station4=pd.read_csv(IN4/'manhattan_canonical_stations.csv',dtype={'station_id':str})
    station4=station4[station4.region=='Midtown'].copy()
    assert len(station4)==190 and station4.station_id.is_unique
    peak4=pd.read_csv(IN4/'manhattan_daily_peak_counts.csv',dtype={'station_id':str},parse_dates=['date'])
    activity4=pd.read_csv(IN4/'manhattan_daily_activity.csv',dtype={'station_id':str},parse_dates=['date'])
    print('Scope: all',len(station4),'historical Midtown analysis IDs; scikit-learn',sklearn.__version__)

    # ### 4.1 Observed station-days and chronological splits
    # A station-day is observed when at least one valid all-day endpoint record exists. If that day has no morning events, the morning count can be zero. A completely unobserved day remains unknown. This avoids assuming closures have zero demand, but misses genuinely operating stations with zero all-day trips. The estimand is therefore demand conditional on activity observation, not unconditional latent demand.
    #
    # | Stage | Dates in 2026 | Purpose |
    # |---|---|---|
    # | Train | July 2–August 31 | Fit both candidates |
    # | Validate | September 1–15 | Select by MAE, separately for P and R |
    # | Test | September 16–29 | Refit on train+validation, then report held-out errors |
    # | Final fit | July 2–September 29 | Generate the five-date static forecast with the selected method |
    #
    # Fit encoders and date scaling on the relevant training data only. Do not shuffle time or input test outcomes, future realized weather or yesterday's realized trips. This is a retrospective date-based evaluation, not a deployment simulation proving that monthly files were available at each historical prediction date.

    observed4=activity4[activity4.station_id.isin(station4.station_id)&activity4.date.dt.dayofweek.lt(5)&activity4.date.between(HISTORY_START4,HISTORY_END4)][['date','station_id']].drop_duplicates()
    wide4=peak4.pivot(index=['date','station_id'],columns='flow',values='count').reset_index()
    panel4=observed4.merge(wide4,on=['date','station_id'],how='left',validate='one_to_one')
    panel4[['P','R']]=panel4[['P','R']].fillna(0)
    panel4['weekday']=panel4.date.dt.dayofweek.astype(str)
    panel4['day_index']=(panel4.date-HISTORY_START4).dt.days.astype(float)
    panel4=panel4.sort_values(['date','station_id']).reset_index(drop=True)
    train4=panel4[panel4.date<=TRAIN_END4].copy()
    val4=panel4[(panel4.date>TRAIN_END4)&(panel4.date<=VAL_END4)].copy()
    test4=panel4[panel4.date>VAL_END4].copy()
    assert train4.date.max()<val4.date.min() and val4.date.max()<test4.date.min()
    assert set(panel4.station_id)==set(station4.station_id)
    coverage4=panel4.groupby('station_id').agg(history_days=('date','nunique'),first_observed=('date','min'),last_observed=('date','max'),weekdays_observed=('weekday','nunique')).reset_index()
    coverage4['training_days']=coverage4.station_id.map(train4.groupby('station_id').size()).fillna(0).astype(int)
    coverage4['days_since_last_observation']=(HISTORY_END4-coverage4.last_observed).dt.days
    coverage4['short_history']=coverage4.history_days<MIN_HISTORY_DAYS4
    coverage4['stale_history']=coverage4.days_since_last_observation>14
    coverage4['needs_extra_review']=coverage4.short_history|coverage4.stale_history|coverage4.weekdays_observed.lt(5)
    split4=pd.DataFrame([{'split':name,'dates':d.date.nunique(),'station_ids':d.station_id.nunique(),'station_days':len(d),'start':d.date.min().date(),'end':d.date.max().date()} for name,d in [('train',train4),('validation',val4),('test',test4)]])
    print(split4)
    print(coverage4[coverage4.needs_extra_review].merge(station4[['station_id','name']],on='station_id'))
    print('Unknown station-days (not filled with zero):',len(station4)*len(pd.bdate_range(HISTORY_START4,HISTORY_END4))-len(panel4))

    # ### 4.2 Two simple methods
    # **Baseline: station-weekday historical means.** Fall back to the station mean when its weekday sample is missing. For unseen stations, fall back to the Midtown weekday mean and flag cold starts separately.
    #
    # **Candidate: regularized Poisson regression.** Fit one shared pickup model and one return model. Features are station ID, weekday and a date trend. One-hot encode station/weekday and standardize dates within training. The mean is `exp(intercept + station effect + weekday effect + date trend)`. Positive predictions and weak regularization limit unstable estimates; no complex station-specific interactions are added.
    #
    # Fix `alpha=0.001` without hyperparameter search. Poisson loss estimates a mean; it does not establish equal real-world mean/variance or calibrated probability intervals. [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
    #
    # Select using validation MAE on previously seen stations only. Report cold starts separately and do not reselect using test outcomes.

    FEATURES4=['station_id','weekday','day_index']
    def fit_mean4(d,flow):
        return {'station_weekday':d.groupby(['station_id','weekday'])[flow].mean().to_dict(),
                'station':d.groupby('station_id')[flow].mean().to_dict(),
                'weekday':d.groupby('weekday')[flow].mean().to_dict(),'global':float(d[flow].mean())}

    def predict_mean4(model,d):
        values=[]
        sources=[]
        for sid,dow in zip(d.station_id,d.weekday):
            if (sid,dow) in model['station_weekday']:
                values.append(model['station_weekday'][(sid,dow)])
                sources.append('station_weekday')
            elif sid in model['station']:
                values.append(model['station'][sid])
                sources.append('station_mean_missing_weekday')
            else:
                values.append(model['weekday'].get(dow,model['global']))
                sources.append('network_weekday_unseen_station')
        return np.asarray(values),np.asarray(sources)

    def fit_poisson4(d,flow):
        transform4=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'),['station_id','weekday']),('date',StandardScaler(),['day_index'])])
        model=Pipeline([('features',transform4),('regression',PoissonRegressor(alpha=ALPHA4,max_iter=1500,tol=1e-7))])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            with threadpool_limits(limits=2):
                model.fit(d[FEATURES4],d[flow])
        return model

    def score_rows4(rows,split,flow,model_name):
        result=[]
        for label,part in [('all_observed',rows),('known_station',rows[rows.known_station]),('unseen_station',rows[~rows.known_station])]:
            if part.empty:
                continue
            denominator=part.actual.sum()
            result.append({'split':split,'flow':flow,'model':model_name,'cohort':label,'station_days':len(part),'station_ids':part.station_id.nunique(),
                           'MAE':mean_absolute_error(part.actual,part.prediction),
                           'WAPE':np.abs(part.actual-part.prediction).sum()/denominator if denominator else np.nan,
                           'bias':float((part.prediction-part.actual).mean())})
        return result

    metrics4=[]
    prediction_rows4=[]
    chosen4={}
    selection_rows4=[]
    # Validation stage: the test block below is not touched during this loop.
    for flow in ['P','R']:
        mean_model=fit_mean4(train4,flow)
        poisson_model=fit_poisson4(train4,flow)
        baseline,source=predict_mean4(mean_model,val4)
        predictions={'weekday_mean':baseline,'poisson':poisson_model.predict(val4[FEATURES4])}
        known=val4.station_id.isin(train4.station_id.unique())
        errors={name:mean_absolute_error(val4.loc[known,flow],pred[known.to_numpy()]) for name,pred in predictions.items()}
        chosen4[flow]='poisson' if errors['poisson']<errors['weekday_mean'] else 'weekday_mean'
        selection_rows4.append({'flow':flow,'selected_model':chosen4[flow],'baseline_validation_MAE':errors['weekday_mean'],'poisson_validation_MAE':errors['poisson']})
        for name,pred in predictions.items():
            r=val4[['date','station_id']].copy()
            r['actual']=val4[flow]
            r['prediction']=pred
            r['known_station']=known
            r['flow']=flow
            r['model']=name
            r['split']='validation'
            metrics4.extend(score_rows4(r,'validation',flow,name))
            prediction_rows4.append(r)
    # Freeze the choices before evaluating the held-out test period.
    (OUT4/'frozen_model_selection.json').write_text(json.dumps(chosen4,indent=2),encoding='utf-8')
    trainval4=pd.concat([train4,val4],ignore_index=True)
    for flow in ['P','R']:
        mean_model=fit_mean4(trainval4,flow)
        poisson_model=fit_poisson4(trainval4,flow)
        baseline,source=predict_mean4(mean_model,test4)
        predictions={'weekday_mean':baseline,'poisson':poisson_model.predict(test4[FEATURES4])}
        known=test4.station_id.isin(trainval4.station_id.unique())
        for name,pred in predictions.items():
            r=test4[['date','station_id']].copy()
            r['actual']=test4[flow]
            r['prediction']=pred
            r['known_station']=known
            r['flow']=flow
            r['model']=name
            r['split']='test'
            assert np.isfinite(pred).all() and (pred>=0).all()
            metrics4.extend(score_rows4(r,'test',flow,name))
            prediction_rows4.append(r)
    metrics4=pd.DataFrame(metrics4)
    prediction4=pd.concat(prediction_rows4,ignore_index=True)
    selection4=pd.DataFrame(selection_rows4)
    metrics4.to_csv(OUT4/'model_metrics.csv',index=False)
    selection4.to_csv(OUT4/'model_selection.csv',index=False)
    print(selection4.round(3))
    print(metrics4[metrics4.cohort=='known_station'].round(3))
    print('MAE is measured in events/station/weekday morning peak. WAPE is total absolute error divided by total actual events, not accuracy. Bias is mean prediction minus actual; positive values indicate overprediction. Test results do not reselect the method.')

    # ### 4.3 Validation and held-out test results
    # The first chart compares candidate methods; the second shows the frozen method on the test period. Daily totals include only observed station-days belonging to previously seen stations. Membership can change by date, so these are not fixed-roster total Midtown demand. Export station-level MAE to avoid hiding errors that cancel in aggregate.

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,flow in zip(axes,['P','R']):
        q=metrics4[(metrics4.flow==flow)&(metrics4.cohort=='known_station')].pivot(index='split',columns='model',values='MAE').reindex(['validation','test'])
        q[['weekday_mean','poisson']].plot.bar(ax=ax,rot=0,color=['#718096','#e69f00'])
        ax.set_ylim(0,float(q.max().max())*1.35)
        ax.set_title(('Pickups' if flow=='P' else 'Returns')+' — MAE')
        ax.set_ylabel('Trips / station / weekday morning')
        ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig(OUT4/'01_model_comparison.png',dpi=150)
    plt.close(fig)
    print('Select methods using validation errors; test data assess generalization only. A more complex method need not improve performance. Use the mean baseline when it wins validation.')
    selected_test4=prediction4[(prediction4.split=='test')&prediction4.known_station&prediction4.apply(lambda r:r.model==chosen4[r.flow],axis=1)].copy()
    selected_test4['abs_error']=(selected_test4.prediction-selected_test4.actual).abs()
    station_errors4=selected_test4.groupby(['station_id','flow']).agg(test_MAE=('abs_error','mean'),test_days=('date','nunique'),mean_actual=('actual','mean')).reset_index()
    daily_test4=selected_test4.groupby(['flow','date']).agg(actual=('actual','sum'),prediction=('prediction','sum'),station_ids=('station_id','nunique')).reset_index()
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,flow in zip(axes,['P','R']):
        q=daily_test4[daily_test4.flow==flow]
        ax.plot(q.date,q.actual,marker='o',label='Actual matched observations')
        ax.plot(q.date,q.prediction,marker='o',label='Selected model')
        ax.set_title(('Pickups' if flow=='P' else 'Returns')+' — held-out weekdays')
        ax.set_ylabel('Trips')
        ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUT4/'02_test_actual_prediction.png',dpi=150)
    plt.close(fig)
    print('The curves show test-period variation and systematic over/underprediction. Similar totals do not imply accurate station forecasts; also examine station MAE. Weather, events and closures can still cause errors.')

    # ### 4.4 Final static demand: two values per station
    # After model selection, refit on all usable history and predict October 29/30 and November 2/3/4. Average each flow separately over these five dates to obtain P_i and R_i.
    #
    # Poisson extrapolates a date trend by approximately one month; it is not a weather or seasonality forecast. The baseline assumes weekday patterns persist. Retain estimates from both candidates for comparison, but use validation to determine the selected value.
    #
    # Keep all 190 IDs. Flag fewer than 20 historical weekdays, no observation in the last 14 historical days, or missing weekday coverage for extra review. These are transparent review rules, not confidence guarantees. Future operating status remains unverified; these conditional demand estimates are not yet complete MILP inputs.

    def predict_demand_candidates(features, mean_model, poisson_model):
        weekday_mean, baseline_source = predict_mean4(mean_model, features)
        poisson = poisson_model.predict(features[FEATURES4])
        return weekday_mean, baseline_source, poisson

    future4=pd.MultiIndex.from_product([FUTURE_DATES4,station4.station_id],names=['date','station_id']).to_frame(index=False)
    future4['weekday']=future4.date.dt.dayofweek.astype(str)
    future4['day_index']=(future4.date-HISTORY_START4).dt.days.astype(float)
    final_daily4=[]
    coefficients4=[]
    final_models4={}
    for flow in ['P','R']:
        mean_model=fit_mean4(panel4,flow)
        poisson_model=fit_poisson4(panel4,flow)
        final_models4[flow]=(mean_model,poisson_model)
        baseline, source, poisson = predict_demand_candidates(future4, mean_model, poisson_model)
        q=future4[['date','station_id','weekday']].copy()
        q['flow']=flow
        q['weekday_mean']=baseline
        q['baseline_source']=source
        q['poisson']=poisson
        q['selected_model']=chosen4[flow]
        q['prediction']=q[chosen4[flow]]
        assert np.isfinite(q[['weekday_mean','poisson','prediction']].to_numpy()).all() and q[['weekday_mean','poisson','prediction']].ge(0).all().all()
        final_daily4.append(q)
        names=poisson_model.named_steps['features'].get_feature_names_out()
        coef=poisson_model.named_steps['regression'].coef_
        coefficients4.append(pd.DataFrame({'flow':flow,'feature':names,'coefficient':coef}))
    final_daily4=pd.concat(final_daily4,ignore_index=True)
    static4=final_daily4.groupby(['station_id','flow']).prediction.mean().unstack('flow').rename(columns={'P':'P_i','R':'R_i'}).reset_index()
    static4=station4[['station_id','name','region','lat','lon']].merge(static4,on='station_id',validate='one_to_one').merge(coverage4,on='station_id',validate='one_to_one')
    for flow in ['P','R']:
        candidates=final_daily4[final_daily4.flow==flow].groupby('station_id')[['weekday_mean','poisson']].mean()
        static4[f'{flow}_baseline']=static4.station_id.map(candidates.weekday_mean)
        static4[f'{flow}_poisson']=static4.station_id.map(candidates.poisson)
        static4[f'{flow}_selected_model']=chosen4[flow]
    static4['demand_unit']='events/station/weekday morning peak'
    static4['service_window']='06:00–10:00 America/New_York'
    static4['forecast_start']=str(FUTURE_DATES4.min().date())
    static4['forecast_end']=str(FUTURE_DATES4.max().date())
    static4['forecast_weekdays']=5
    static4['holidays_excluded']=False
    static4['future_operating_status_verified']=False
    static4['scenario_assumption']='conditional_on_station_operating; completed-trip demand proxy'
    static4['quality_note']=np.where(static4.needs_extra_review,'Short/stale history or incomplete weekday coverage: extra review required.','More complete history; future operating status remains unverified.')
    assert len(static4)==190 and static4.station_id.is_unique and static4[['P_i','R_i']].notna().all().all()
    static4.to_csv(OUT4/'midtown_static_demand.csv',index=False,encoding='utf-8-sig')
    print(static4[['station_id','name','P_i','R_i','history_days','last_observed','quality_note']].round(1).head(15))
    print('Exported all',len(static4),'IDs; extra review:',int(static4.needs_extra_review.sum()))
    print('Forecast dates:',[str(d.date()) for d in FUTURE_DATES4])

    # ### 4.5 Demand patterns across stations
    # The x-axis is estimated pickups and the y-axis estimated returns. Above the diagonal means more returns; below means more pickups. Neither direction establishes surplus or shortage without initial stock, capacity and a target.

    fig,ax=plt.subplots(figsize=(7,6))
    for flag,label,color in [(False,'More history; status unverified','#2878a5'),(True,'Extra history/status review','#d95f02')]:
        d=static4[static4.needs_extra_review==flag]
        ax.scatter(d.P_i,d.R_i,s=26,alpha=.7,label=label,color=color)
    limit=max(static4.P_i.max(),static4.R_i.max())*1.05
    ax.plot([0,limit],[0,limit],'k--',lw=1,label='P = R')
    ax.set_xlim(0,limit)
    ax.set_ylim(0,limit)
    ax.set_xlabel('Predicted pickups / weekday morning')
    ax.set_ylabel('Predicted returns / weekday morning')
    ax.set_title('Midtown: all 190 historical analysis IDs')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT4/'03_static_demand.png',dpi=150)
    plt.close(fig)
    print('Each point is a retained historical analysis ID. Orange marks extra review, not exclusion. The plot checks demand scale and pickup/return differences; it does not establish truck tasks.')
    selected_metrics4=metrics4[(metrics4.split=='test')&(metrics4.cohort=='known_station')&metrics4.apply(lambda r:r.model==chosen4[r.flow],axis=1)]
    print(selected_metrics4[['flow','model','MAE','WAPE','bias','station_ids','station_days']].round(3))
    summary4={'region':'Midtown','historical_station_ids':len(static4),'selected_models':chosen4,'forecast_dates':[str(d.date()) for d in FUTURE_DATES4],
              'extra_review_station_ids':static4.loc[static4.needs_extra_review,'station_id'].tolist(),'test_is_retrospective':True,'future_status_verified':False,'inventory_parameters_created':False,'optimization_run':False}
    (OUT4/'stage4_summary.json').write_text(json.dumps(summary4,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT4/'environment.json').write_text(json.dumps({'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},indent=2),encoding='utf-8')
    print('Step 4 complete: demand estimates are available, but I0, C, I_star and G are not yet prepared; MILP inputs remain incomplete.')

    # ### Interpreting Step 4
    # Validation pickup MAE is **12.407** for Poisson versus **12.543** for the baseline, an improvement of about **1.1%**. For returns, the baseline MAE **15.554** is slightly better than Poisson **15.720**. Freeze **Poisson for P and weekday means for R**.
    #
    # | Flow | Selected method | Test MAE | Test WAPE |
    # |---|---|---:|---:|
    # | Pickups P | Poisson | 11.910 | 22.1% |
    # | Returns R | Weekday mean | 15.027 | 21.8% |
    #
    # Test evaluation covers 183 previously seen IDs and 1,710 observed station-days. MAE units are events/station/weekday morning peak. Poisson return test MAE is lower at 13.965, showing that the small validation difference is unstable. Do not switch methods using the same test set and still call it independent; a revised method needs a new evaluation window.
    #
    # All **190 historical IDs** remain in the forecast output, with **7** requiring extra history review. One test-period cold-start ID is reported separately. The evaluation count of 183 is not a reduction in study scope.
    #
    # Use `outputs_stage4/midtown_static_demand.csv` for conditional P_i/R_i estimates with dates, units, method and quality flags. WAPE is not accuracy. September performance does not verify late-October predictions or current station operations. Before optimization, add case-date station status, capacity and midnight inventory.

    # ### Output location
    # Step 4 exports to `outputs_stage4/`. The next section prepares a same-day inventory case; keep the five-day average and case-date predictions distinct.

    # ## Step 5: Midnight inventory and case-day demand

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone
    import re
    from decimal import Decimal

    CASE_DATE5=pd.Timestamp('2026-10-28')
    MIDNIGHT5=CASE_DATE5.tz_localize('America/New_York')
    TARGET_TS5=int(MIDNIGHT5.timestamp())
    OUT5=Path.cwd()/'outputs_stage5'
    OUT5.mkdir(exist_ok=True)
    CACHE5=Path.cwd()/'data/stage5'
    CACHE5.mkdir(parents=True,exist_ok=True)
    ARCHIVE5=CACHE5/str(CASE_DATE5.date())
    ARCHIVE5.mkdir(exist_ok=True)
    INFO_URL5='https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json'
    ARCHIVE_URL5='https://ctbk-gbfs-api.hccs-ctbk.workers.dev/api/stations'
    # Reuse the fixed snapshot; do not silently refresh it to a different date.
    info_path5=CACHE5/'station_information_snapshot.json'
    if not info_path5.exists():
        r=requests.get(INFO_URL5,timeout=30)
        r.raise_for_status()
        payload=r.json()
        payload['_retrieved_at_utc']=datetime.now(timezone.utc).isoformat()
        info_path5.write_text(json.dumps(payload),encoding='utf-8')
    info5=json.loads(info_path5.read_text())
    meta5=pd.DataFrame(info5['data']['stations']).rename(columns={'station_id':'gbfs_uuid','short_name':'gbfs_short_name','name':'gbfs_name','lat':'gbfs_lat','lon':'gbfs_lon','capacity':'physical_capacity_snapshot'})
    meta5['gbfs_short_name']=meta5.gbfs_short_name.astype(str)
    def id_key5(value):
        value=str(value)
        return format(Decimal(value).normalize(),'f') if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',value) else value
    meta5['id_key']=meta5.gbfs_short_name.map(id_key5)
    x5,y5=project(meta5.gbfs_lon.to_numpy(),meta5.gbfs_lat.to_numpy())
    meta5['in_midtown_snapshot']=[island.covers(Point(x,y)) for x,y in zip(x5,y5)]
    meta5['in_midtown_snapshot'] &= (y5>=boundary_y(x5,14))&(y5<boundary_y(x5,59))
    meta5['near_split_30m']=[min(boundaries[14].distance(Point(x,y)),boundaries[59].distance(Point(x,y)))<=30 for x,y in zip(x5,y5)]
    # Do not arbitrarily choose a UUID when mappings are duplicated.
    meta5['duplicate_key']=meta5.id_key.duplicated(keep=False)
    assert not meta5.loc[meta5.in_midtown_snapshot,'duplicate_key'].any(), 'Duplicate ID mapping in the Midtown roster; review before continuing.'
    hist5=static4[['station_id','name','region','lat','lon','history_days','needs_extra_review']].copy()
    hist5['id_key']=hist5.station_id.map(id_key5)
    hist5['has_historical_prediction']=True
    keep5=['id_key','gbfs_uuid','gbfs_short_name','gbfs_name','gbfs_lat','gbfs_lon','physical_capacity_snapshot','in_midtown_snapshot','near_split_30m']
    roster5=hist5.merge(meta5.loc[~meta5.duplicate_key,keep5],on='id_key',how='left',validate='one_to_one')
    new5=meta5[meta5.in_midtown_snapshot&~meta5.id_key.isin(hist5.id_key)].copy()
    new5=new5[~new5.duplicate_key].copy()
    new5['station_id']=new5.gbfs_short_name
    new5['name']=new5.gbfs_name
    new5['region']='Midtown'
    new5['lat']=new5.gbfs_lat
    new5['lon']=new5.gbfs_lon
    new5['history_days']=0
    new5['needs_extra_review']=True
    new5['has_historical_prediction']=False
    roster5=pd.concat([roster5,new5[roster5.columns]],ignore_index=True)
    roster5['mapping_distance_m']=np.hypot((roster5.lat-roster5.gbfs_lat)*111195,(roster5.lon-roster5.gbfs_lon)*111195*np.cos(np.deg2rad(roster5.lat)))
    # Normalize whitespace and the verified Central Park South/S variant; preserve ID suffixes.
    def name_key5(value):
        return re.sub(r'\s+',' ',str(value).strip().casefold()).replace('central park south','central park s')
    roster5['name_matches']=roster5.name.map(name_key5).eq(roster5.gbfs_name.map(name_key5))
    roster5['mapping_ok']=roster5.gbfs_uuid.notna()&roster5.mapping_distance_m.le(50)&roster5.name_matches&roster5.in_midtown_snapshot.fillna(False).astype(bool)
    roster5['metadata_snapshot_time']=pd.to_datetime(info5['last_updated'],unit='s',utc=True).tz_convert('America/New_York').isoformat()
    print('Historical IDs:',len(hist5),'; Midtown metadata IDs:',int(meta5.in_midtown_snapshot.sum()),'; union review roster:',len(roster5))
    print(roster5[~roster5.mapping_ok|~roster5.has_historical_prediction][['station_id','name','gbfs_name','has_historical_prediction','mapping_ok']])

    # ### 5.1 Read real midnight observations and retain missingness reasons
    # Query the 15 minutes before midnight. Select the last observation whose `ts`, `polled_at` and `last_reported` are not later than midnight. Require observation/poll age <=300 seconds and station-report age <=900 seconds, plus installed, renting and returning status. These quality thresholds do not guarantee an exact midnight count.
    #
    # I0 is available bikes, including e-bikes once. Effective capacity is available bikes plus available docks, assuming unavailable bikes/docks do not recover during this case. Keep physical capacity from the metadata snapshot separately; the two measures are not interchangeable.
    #
    # Sources: [ctbk](https://ctbk.dev/stations), [official station information](https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json), [GBFS field reference](https://gbfs.org/documentation/reference/). Retain snapshots, responses and request logs.
    #
    # Normalize whitespace and the verified Central Park South/S variant only. Match IDs and require coordinates within 50 m. If effective capacity exceeds the metadata physical capacity, block target generation pending review. Passing this check does not prove historical capacity was unchanged.

    def fetch_midnight5(uuid):
        path=ARCHIVE5/f'{uuid}.json'
        url=f'{ARCHIVE_URL5}/{uuid}/range'
        params={'from':TARGET_TS5-900,'to':TARGET_TS5}
        if path.exists():
            return uuid,json.loads(path.read_text()),'cached'
        try:
            response=requests.get(url,params=params,timeout=25)
            response.raise_for_status()
            payload=response.json()
            if not isinstance(payload.get('rows'),list):
                raise ValueError('archive rows missing')
            payload['_request_url']=response.url
            payload['_retrieved_at_utc']=datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(payload),encoding='utf-8')
            return uuid,payload,'downloaded'
        except Exception as error:
            return uuid,None,type(error).__name__+': '+str(error)[:160]

    archives5={}
    request_log5=[]
    uuids5=roster5.gbfs_uuid.dropna().unique().tolist()
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(fetch_midnight5,uuid) for uuid in uuids5]
        for completed,job in enumerate(as_completed(jobs),1):
            uuid,payload,status=job.result()
            archives5[uuid]=payload
            request_log5.append({'gbfs_uuid':uuid,'result':status,'rows':len(payload['rows']) if payload else 0})
            if completed%40==0:
                print('Checked',completed,'/',len(jobs),flush=True)
    pd.DataFrame(request_log5).to_csv(OUT5/'archive_request_log.csv',index=False)

    records5=[]
    for row in roster5.itertuples():
        rec={'station_id':row.station_id,'inventory_status':'unmatched_station','I0':np.nan,'C_effective':np.nan}
        payload=archives5.get(row.gbfs_uuid)
        if pd.notna(row.gbfs_uuid):
            rec['inventory_status']='request_failed' if payload is None else 'no_prior_snapshot'
        if payload is not None:
            rec['archive_capacity_reference']=payload.get('capacity') # Unverified timestamp context; not proof of midnight capacity.
            d=pd.DataFrame(payload['rows'])
            if not d.empty:
                required=['ts','polled_at','last_reported','num_bikes_available','num_docks_available','is_installed','is_renting','is_returning']
                if not set(required).issubset(d.columns):
                    rec['inventory_status']='missing_required_field'
                else:
                    d[required]=d[required].apply(pd.to_numeric,errors='coerce')
                    eligible=d[required[:3]].notna().all(axis=1)&d[required[:3]].le(TARGET_TS5).all(axis=1)
                    if 'station_id' in d:
                        eligible &= d.station_id.eq(row.gbfs_uuid)
                    d=d[eligible].sort_values(['ts','polled_at'])
                    if len(d):
                        s=d.iloc[-1]
                        rec.update({k:s.get(k,np.nan) for k in required+['num_ebikes_available','num_bikes_disabled','num_docks_disabled']})
                        rec['snapshot_age_seconds']=TARGET_TS5-s.ts
                        rec['polled_age_seconds']=TARGET_TS5-s.polled_at
                        rec['report_age_seconds']=TARGET_TS5-s.last_reported
                        counts=np.array([s.num_bikes_available,s.num_docks_available],dtype=float)
                        fresh=(rec['snapshot_age_seconds']<=300 and rec['polled_age_seconds']<=300 and rec['report_age_seconds']<=900)
                        valid=np.isfinite(counts).all() and (counts>=0).all() and (counts==np.floor(counts)).all()
                        operating=all(s[k]==1 for k in ['is_installed','is_renting','is_returning'])
                        rec['inventory_status']='valid' if fresh and valid and operating else ('stale_snapshot' if not fresh else 'invalid_counts' if not valid else 'not_operating')
                        if rec['inventory_status']=='valid':
                            rec['I0']=int(s.num_bikes_available)
                            rec['C_effective']=int(counts.sum())
                            if rec['C_effective']==0:
                                rec['inventory_status']='zero_effective_capacity'
        records5.append(rec)
    inventory5=pd.DataFrame(records5)
    inventory5['case_midnight']=MIDNIGHT5.isoformat()
    inventory5.to_csv(OUT5/'midnight_inventory_audit.csv',index=False,encoding='utf-8-sig')
    print(inventory5.inventory_status.value_counts().rename_axis('inventory_status').to_frame('station_ids'))

    # ### 5.2 Attach same-day pickup and return estimates

    case_features5=station4[['station_id']].copy()
    case_features5['date']=CASE_DATE5
    case_features5['weekday']=str(CASE_DATE5.dayofweek)
    case_features5['day_index']=float((CASE_DATE5-HISTORY_START4).days)
    assert CASE_DATE5>HISTORY_END4
    for flow in ['P','R']:
        mean_model,poisson_model=final_models4[flow]
        weekday_mean, _, poisson = predict_demand_candidates(case_features5, mean_model, poisson_model)
        case_features5[f'{flow}_case'] = poisson if chosen4[flow] == 'poisson' else weekday_mean
        case_features5[flow+'_model']=chosen4[flow]
    case_features5.to_csv(OUT5/'case_day_demand.csv',index=False)
    params5=roster5.merge(inventory5,on='station_id',validate='one_to_one').merge(case_features5[['station_id','P_case','R_case','P_model','R_model']],on='station_id',how='left',validate='one_to_one')
    params5['capacity_exceeds_snapshot']=params5.C_effective.gt(params5.physical_capacity_snapshot)
    params5['data_complete']=~params5.capacity_exceeds_snapshot&params5.mapping_ok&params5.inventory_status.eq('valid')&params5[['P_case','R_case']].notna().all(axis=1)
    params5['needs_history_review']=params5.needs_extra_review.astype(bool)
    params5['case_date'] = str(CASE_DATE5.date())
    print(f'Midtown roster: {len(params5)} records')

    # ## Step 6: Retain eligible stations and allocate fleet-conserving targets

    from pathlib import Path
    import csv, json, math
    OUT6=Path('outputs_stage6')
    OUT6.mkdir(exist_ok=True)
    rows6=params5.to_dict('records')
    def flag6(row,key):
        return str(row.get(key,'')).lower()=='true'
    def number6(row,key):
        try:
            return float(row[key])
        except (ValueError,KeyError):
            return float('nan')
    accepted6=[]
    excluded6=[]
    for r in rows6:
        reasons=[]
        if not flag6(r,'mapping_ok'):
            reasons.append('Station identity or location unverified')
        if not flag6(r,'has_historical_prediction'):
            reasons.append('No historical demand model')
        if r['inventory_status']!='valid':
            reasons.append('Inventory status: '+r['inventory_status'])
        if flag6(r,'capacity_exceeds_snapshot'):
            reasons.append('Effective capacity exceeds metadata capacity')
        if flag6(r,'needs_history_review'):
            reasons.append('Short/stale history or incomplete weekday coverage')
        vals={k:number6(r,k) for k in ['I0','C_effective','P_case','R_case','lat','lon']}
        if not all(math.isfinite(v) for v in vals.values()):
            reasons.append('Missing or non-finite key parameters')
        if not reasons:
            C,I0,P,R=[vals[k] for k in ['C_effective','I0','P_case','R_case']]
            assert C>0 and 0<=I0<=C and P>=0 and R>=0
            assert C==int(C) and I0==int(I0)
            assert r['case_date']=='2026-10-28'
            accepted6.append({'id':r['station_id'],'name':r['name'],'lat':vals['lat'],'lon':vals['lon'],
              'C':int(C),'I0':int(I0),'P':P,'R':R,
              'case_date':r['case_date'],'demand_window':'06:00-10:00 America/New_York',
              'capacity_basis':'midnight available bikes + available docks'})
        else:
            excluded6.append({'station_id':r['station_id'],'name':r['name'],'exclusion_reasons':'; '.join(reasons)})
    assert len({r['station_id'] for r in rows6})==len(rows6)
    assert len(accepted6)+len(excluded6)==len(rows6)
    assert not ({r['id'] for r in accepted6}&{r['station_id'] for r in excluded6})
    pd.DataFrame(excluded6, columns=['station_id','name','exclusion_reasons']).to_csv(OUT6/'excluded_stations.csv', index=False)
    print(f'Retained {len(accepted6)} stations, excluded {len(excluded6)}.')

    # ### Fleet-conserving target inventory
    #
    # Replay historical 06:00–10:00 events, scaling pickup and return flows to the case-day forecasts. Allocate integer target inventories within station capacities while preserving the total midnight fleet. The replay loss counts unmet pickup and return events in the simulation.

    from pathlib import Path
    cache = Path("data/target_inventory")

    if not all((cache / name).is_file() for name in (
        "september_peak_events.csv.gz",
        "observed_station_days.csv",
        "source_manifest.json",
    )): 
        import pandas as pd, json, zipfile, csv, gzip, hashlib
        from pathlib import Path
        out=Path('data/target_inventory')
        out.mkdir(parents=True,exist_ok=True)
        ids={s['id'] for s in accepted6}
        a=pd.read_csv('outputs_stage2/alias_resolution.csv',dtype=str)
        aliases=dict(zip(a.station_id,a.canonical_station_id))
        parts=[]
        seen=set()
        total=0
        dups=0
        with zipfile.ZipFile('data/raw/202609-citibike-tripdata.zip') as z:
         for name in z.namelist():
          if not name.endswith('.csv') or name.startswith('__MACOSX/'):
              continue
          print('Reading',name,flush=True)
          with z.open(name) as f:
           for chunk in pd.read_csv(f,usecols=['ride_id','started_at','ended_at','start_station_id','end_station_id'],dtype=str,chunksize=200000):
            total+=len(chunk)
            chunk=chunk.dropna(subset=['ride_id'])
            dup=chunk.ride_id.duplicated()|chunk.ride_id.isin(seen)
            dups+=int(dup.sum())
            seen.update(chunk.ride_id)
            chunk=chunk.loc[~dup]
            for side,col,delta in [('start','started_at',-1),('end','ended_at',1)]:
             sid=chunk[side+'_station_id'].str.strip()
             sid=sid.map(aliases).fillna(sid)
             mask=sid.isin(ids)
             t=pd.to_datetime(chunk.loc[mask,col],errors='coerce',format='mixed')
             keep=t.notna()&(t.dt.dayofweek<5)&(t.dt.hour>=6)&(t.dt.hour<10)&(t.dt.normalize().between('2026-09-01','2026-09-29'))
             idx=t.index[keep]
             times=t.loc[idx]
             parts.append(pd.DataFrame({'station_id':sid.loc[idx],'date':times.dt.strftime('%Y-%m-%d'),
                   'time':times.dt.strftime('%H:%M:%S.%f'),'delta':delta}))
        events=pd.concat(parts,ignore_index=True).sort_values(['station_id','date','time','delta'],ascending=[True,True,True,False])
        activity=pd.read_csv('outputs_stage2/manhattan_daily_activity.csv',dtype={'station_id':str},parse_dates=['date'])
        print(activity.columns,flush=True)
        days=activity[activity.station_id.isin(ids)&activity.date.between('2026-09-01','2026-09-29')&(activity.date.dt.dayofweek<5)][['station_id','date']].drop_duplicates()
        days['date']=days.date.dt.strftime('%Y-%m-%d')
        events.to_csv(out/'september_peak_events.csv.gz',index=False,compression='gzip')
        days.to_csv(out/'observed_station_days.csv',index=False)
        manifest={'source':'https://tripdata.s3.amazonaws.com/202609-citibike-tripdata.zip',
         'source_zip_bytes':Path('data/raw/202609-citibike-tripdata.zip').stat().st_size,'rows_read':total,'duplicate_ride_ids':dups,
         'events':len(events),'observed_station_days':len(days),'start':'2026-09-01','end':'2026-09-29',
         'window':'06:00-10:00 America/New_York; Monday-Friday including holidays',
         'equal_timestamp_rule':'returns before pickups (arbitrary tie convention)',
         'event_cache_sha256':hashlib.sha256((out/'september_peak_events.csv.gz').read_bytes()).hexdigest()}
        (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(manifest,flush=True)

    # #### 6.1 Score each starting inventory
    #
    # For each station, replay the same morning events starting with 0 bikes, 1 bike, and so on up to capacity. Save the average blocked pickups and returns for each starting inventory.

    # Compute a replay loss for each possible starting inventory.
    from pathlib import Path
    import json, csv
    import numpy as np
    import pandas as pd

    target_out=Path('outputs_stage6');target_out.mkdir(exist_ok=True)
    target_stations=sorted(accepted6,key=lambda s:s['id'])
    target_events=pd.read_csv('data/target_inventory/september_peak_events.csv.gz',dtype={'station_id':str})
    target_days=pd.read_csv('data/target_inventory/observed_station_days.csv',dtype={'station_id':str})
    target_source=json.loads(Path('data/target_inventory/source_manifest.json').read_text())
    target_events=target_events.sort_values(['station_id','date','time','delta'],ascending=[True,True,True,False])
    ids = [str(s["id"]) for s in target_stations]
    assert ids and len(ids) == len(set(ids)), "Expected nonempty, unique station IDs"
    assert all(s['case_date']=='2026-10-28' for s in target_stations)
    assert target_events.date.max() < '2026-10-28'
    station_events={sid:g for sid,g in target_events.groupby('station_id')}
    station_dates={sid:list(g.date) for sid,g in target_days.groupby('station_id')}
    cost_curves = []
    curve_rows = []
    coverage_rows = []
    for s in target_stations:
        sid = s['id']
        capacity = int(s['C'])
        dates = station_dates.get(sid, [])
        if not dates:raise ValueError(f'{sid} has no historical weekdays to replay; stop rather than invent sequences.')
        group=station_events.get(sid,pd.DataFrame(columns=['date','delta']))
        mean_p=float((group.delta<0).sum())/len(dates)
        mean_r=float((group.delta>0).sum())/len(dates)
        # Keep case-day P/R forecasts; historical sequences supply order and scaled mean flow totals.
        if (mean_p==0 and s['P']>0) or (mean_r==0 and s['R']>0):
            raise ValueError(f'{sid} has positive predicted flow without historical events in that direction; scaling is unsupported.')
        p_weight=s['P']/mean_p if mean_p else 0
        r_weight=s['R']/mean_r if mean_r else 0
        day_sequences={d:g.delta.to_numpy() for d,g in group.groupby('date')}
        losses=np.zeros(capacity+1)
        for date in dates:
            stock=np.arange(capacity+1,dtype=float)
            for delta in day_sequences.get(date,[]):
                proposed=stock+(r_weight if delta>0 else -p_weight)
                losses+=np.maximum(-proposed,0)+np.maximum(proposed-capacity,0)
                stock=np.clip(proposed,0,capacity)
        losses/=len(dates)
        cost_curves.append(losses)
        for b, value in enumerate(losses):
            curve_rows.append({'station_id':sid,'initial_bikes':b,'mean_simulated_loss':float(value)})
        coverage_rows.append({'station_id':sid,'observed_weekdays':len(dates),'historical_mean_pickups':mean_p,
            'historical_mean_returns':mean_r,'pickup_event_weight':p_weight,'return_event_weight':r_weight})

    # #### 6.2 Allocate the fleet
    #
    # After each station, `dp[b]` stores the lowest replay loss for allocating exactly `b` bikes so far. Try each possible inventory at the next station. Save the winning choice so the final allocation can be recovered by working backward.

    # Dynamic programming chooses integer station targets minimizing replay loss with exact conservation.
    # Within 1e-9 replay-loss tolerance, prefer less total adjustment from initial stocks.
    total_bikes=sum(int(s['I0']) for s in target_stations)
    dp=np.full(total_bikes+1,np.inf);dp[0]=0
    secondary=np.full(total_bikes+1,np.inf);secondary[0]=0
    choices=np.full((len(target_stations),total_bikes+1),-1,dtype=np.int16)
    for i,(s,costs) in enumerate(zip(target_stations,cost_curves)):
        next_dp = np.full_like(dp, np.inf)
        next_secondary = np.full_like(dp, np.inf)
        for bikes,cost in enumerate(costs[:total_bikes+1]):
            candidate=dp[:total_bikes+1-bikes]+cost
            candidate_secondary=secondary[:total_bikes+1-bikes]+abs(bikes-s['I0'])
            old=next_dp[bikes:];old_secondary=next_secondary[bikes:]
            # Compare only reachable allocations. This avoids subtracting infinity.
            reachable = np.isfinite(candidate)
            both_reachable = reachable & np.isfinite(old)
            tie = np.zeros(len(candidate), dtype=bool)
            tie[both_reachable] = abs(candidate[both_reachable] - old[both_reachable]) <= 1e-9

            lower_loss = candidate < old - 1e-9
            less_movement = candidate_secondary < old_secondary
            better = reachable & (lower_loss | (tie & less_movement))
            old[better] = candidate[better]
            old_secondary[better] = candidate_secondary[better]
            choices[i, bikes:][better] = bikes
        dp,secondary=next_dp,next_secondary
    assert np.isfinite(dp[total_bikes]),'No feasible target allocation under fleet-total and capacity constraints.'
    new_targets = []
    remaining = total_bikes
    for i in range(len(target_stations)-1,-1,-1):
        bikes = int(choices[i, remaining])
        assert bikes >= 0
        new_targets.append(bikes)
        remaining -= bikes
    new_targets.reverse()
    assert remaining == 0

    # #### 6.3 Save the routing inputs
    #
    # Each station gets a target `I_star` and an inventory gap `G = I_star - I0`.

    for s,new in zip(target_stations,new_targets):
        s['I_star']=int(new)
        s['G']=int(new-s['I0'])
        s['target_method']='fixed_fleet_scaled_historical_event_replay_v1'
    assert sum(s['I_star'] for s in target_stations)==total_bikes
    assert sum(s['G'] for s in target_stations)==0
    assert all(0<=s['I_star']<=s['C'] for s in target_stations)
    new_loss=sum(float(costs[s['I_star']]) for s,costs in zip(target_stations,cost_curves))
    initial_loss=sum(float(costs[s['I0']]) for s,costs in zip(target_stations,cost_curves))
    assert new_loss<=initial_loss+1e-6,'New targets must not perform worse than the feasible initial-stock allocation in this replay model.'
    target_summary={'case_date':'2026-10-28','target_method':'fixed_fleet_scaled_historical_event_replay_v1',
        'stations':len(target_stations),'initial_inventory_total':total_bikes,
        'target_inventory_total':sum(new_targets),
        'net_target_gap':0,'surplus_stations':sum(s['G']<0 for s in target_stations),
        'deficit_stations':sum(s['G']>0 for s in target_stations),'balanced_stations':sum(s['G']==0 for s in target_stations),
        'surplus_bikes':sum(max(-s['G'],0) for s in target_stations),'deficit_bikes':sum(max(s['G'],0) for s in target_stations),
        'simulated_mean_loss_initial':initial_loss,
        'simulated_mean_loss_new_target':new_loss,'historical_sequence_start':target_source['start'],
        'historical_sequence_end':target_source['end'],'event_cache_sha256':target_source['event_cache_sha256'],
        'independent_service_test':False,'routing_optimization_run':False}
    def save_target_csv(name,rows):
        with (target_out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    save_target_csv('inventory_service_curves.csv',curve_rows)
    save_target_csv('event_replay_coverage.csv',coverage_rows)
    save_target_csv('final_station_parameters.csv',target_stations)
    dataset6={'stations':target_stations}
    (target_out/'model_station_inputs.json').write_text(json.dumps(dataset6,ensure_ascii=False,indent=2))
    (target_out/'stage6_summary.json').write_text(json.dumps(target_summary,ensure_ascii=False,indent=2))
    print(json.dumps(target_summary,ensure_ascii=False,indent=2))
    print('Rerun Step 8; the old solution.json does not correspond to the new targets.')


if __name__ == "__main__":
    main()
