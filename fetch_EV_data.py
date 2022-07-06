
import os
import sys
import json
from tqdm import tqdm
import requests
import numpy as np
from dateutil.parser import parse as parse_date
import datetime    


HTTP_status_codes = {400: 'BAD_REQUEST',
                     401: 'UNAUTHORIZED',
                     403: 'FORBIDDEN',
                     500: 'INTERNAL_SERVER_ERROR',
                     501: 'NOT_IMPLEMENTED'}


default_outfile = 'EV_data.json'
progname = os.path.basename(sys.argv[0])


def usage():
    print(f'usage: {progname} [<options>] config_file')
    print('')
    print(f'    -o, --output          output file name (default: "{default_outfile}")')
    print( '    -f, --force           force overwrite of output file if it exists')
    print( '    -h, --help            print this help message and exit')
    print('')


# A description of the data and the REST API used here can be found at the
# following address: https://ev.caltech.edu/dataset

if __name__ == '__main__':

    if len(sys.argv) == 1:
        usage()
        sys.exit(0)

    i = 1
    n_args = len(sys.argv)
    outfile = None
    force = False

    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-o', '--output-file'):
            outfile = sys.argv[i+1]
            i += 1
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif arg[0] == '-':
            print(f'{progname}: unknown option: `{arg}`')
            sys.exit(1)
        else:
            break
        i += 1

    if i == n_args:
        usage()
        sys.exit(0)

    if i < n_args-1:
        print(f'{progname}: cannot specify options after configuration file name.')
        sys.exit(2)

    config_file = sys.argv[i]
    if not os.path.isfile(config_file):
        print(f'{progname}: {config_file}: no such file.')
        sys.exit(3)
    config = json.load(open(config_file, 'r'))

    if outfile is None:
        try:
            outfile = config['output']
        except:
            outfile = default_outfile
    if os.path.isfile(outfile) and not force:
        print(f'{progname}: {outfile}: file exists, use -f to overwrite it.')
        sys.exit(4)
        
    credentials = os.environ['CALTECH_EV_API_TOKEN'], ''
    base_url = 'https://ev.caltech.edu/api/v1/'
    site_id = config['site_id'] if 'site_id' in config else 'caltech'
    with_ts = config['time_series'] if 'time_series' in config else False
    pretty = config['pretty'] if 'pretty' in config else False
    url = base_url + 'sessions/' + site_id
    if with_ts:
        url += '/ts'

    filters = config['filters']
    logicals = config['logicals']
    if len(logicals) != len(filters) - 1:
        print('The number of logical conditions must be equal to that of filters minus 1')
        sys.exit(5)

    def format_filter(name, op, value):
        if name == 'connectionTime':
            day = parse_date(value)
            value = day.strftime('%a, %-d %b %Y %H:%M:%S GMT')
        if isinstance(value, str):
            return '{}{}"{}"'.format(name, op, value)
        return '{}{}{}'.format(name, op, value)
    
    try:
        url += '?where=' + format_filter(filters[0][0], filters[0][1], filters[0][2])
        for filt, and_or in zip(filters[1:], logicals):
            url += ' {} {}'.format(and_or, format_filter(filt[0], filt[1], filt[2]))
    except:
        print('There are no filters')

    if pretty:
        if '?' not in url:
            url += '?pretty'
        else:
            url += '&pretty'

    if 'sort' in config and config['sort'] is not None:
        url += '&sort=' + config['sort']

    items = []
    n_pages = None

    with tqdm(total=100, unit_scale=True) as pbar:
        while True:

            response = requests.get(url, auth=credentials)            
            if response.status_code != 200:
                print(f'An error occured ({HTTP_status_codes[response.status_code]}) for URL')
                print(url)
                sys.exit(6)

            data = json.loads(response.content.decode(response.encoding))
            items += data['_items']
            if n_pages is None:
                n_pages = int(np.ceil(data['_meta']['total'] / data['_meta']['max_results']))
                if n_pages == 0:
                    print('No data matches the filters.')
                    sys.exit(0)
                step = 100 / n_pages
            pbar.update(step)
            try:
                url = base_url + data['_links']['next']['href']
            except:
                break

    json.dump(items, open(outfile, 'w'), indent=4)

