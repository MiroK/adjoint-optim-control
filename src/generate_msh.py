import os, subprocess, itertools
from numpy import deg2rad


def generate_mesh(args, template, dim=2):
    '''Modify template according args and make gmsh generate the mesh'''
    assert os.path.exists(template), template
    
    args = args.copy()

    with open(template, 'r') as f: old = f.readlines()

    # Chop the file to replace the jet positions
    split = list(map(lambda s: s.startswith('DefineConstant'), old)).index(True)

    # Look for ] closing DefineConstant[
    # Make sure that args specifies all the constants (not necessary
    # as these have default values). This is more to check sanity of inputs
    last, _ = next(itertools.dropwhile(lambda i_line: '];' not in line,
                                       enumerate(old)))
    constant_lines = old[split+1:last]
    constants = set(l.split('=')[0].strip() for l in constant_lines)

    jet_positions = deg2rad([90, 270])
    jet_positions = 'jet_positions[] = {%s};\n' % (', '.join(map(str, jet_positions)))

    output = args.pop('output')

    if not output:
        output = '_'.join([template, 'templateted.geo'])
    assert os.path.splitext(output)[1] == '.geo'

    body = ''.join(old)
    with open(output, 'w') as f:
        f.write(body)

    args['jet_width'] = deg2rad(args['jet_width'])

    scale = args.pop('clscale')

    # What we think can be defined vs what can be
    assert set(args.keys()) <= constants, (set(args.keys())-constants)

    constant_values = ' '.join(['-setnumber %s %g' % item for item in args.items()])

    return subprocess.call(['gmsh -%d -clscale %g %s %s' % (dim, scale, constant_values, output)],
                           shell=True)
