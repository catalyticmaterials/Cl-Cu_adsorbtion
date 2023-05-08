import math
import os
import re
import csv
import argparse
import matplotlib.pyplot as plt
from operator import attrgetter
from itertools import groupby
from typing import Sequence, Callable, Any, Optional
from dataclasses import dataclass

@dataclass
class slab_result:
    file_name: str
    Free_energy: float
    Cell_surface_area: float
    facet: str

def set_slap_object(fil:str) -> slab_result | None:
    try:
        with open(fil, 'r') as workfile:
            work_file_dat = workfile.read()
        FE = float(re.findall(r'Free\senergy:\s+(?P<FE>-?\d+\.?\d+)', work_file_dat)[-1])
        name_format_grouping = re.search(r'(?P<facet>\d{3}(-\dX\d)?)(?:_slab(_per\d)?)', fil)
        unit_cell = re.findall(
                r'(?:1.\saxis:\s+\w+\s+)(?P<x1>\d+\.\d+)(?:.*)(?:\s+2.\saxis:\s+\w+\s+\d+\.\d+\s+)(?P<y2>\d+\.\d+)',
                work_file_dat)[-1]
        return slab_result(file_name=fil,Free_energy=FE,Cell_surface_area=round(float(unit_cell[0]), 5) * round(float(unit_cell[1]), 5),facet=name_format_grouping.group('facet'))
    except: return None

@dataclass
class adsorbate_result:
    file_name: str
    Free_energy: float
    Cell_surface_area: float
    adsorb_type: str

def set_adsorbate_object(fil:str) -> adsorbate_result | None:
    try:
        with open(fil, 'r') as workfile:
            work_file_dat = workfile.read()
        adsorbat_match = re.search(r'(?P<facet>\d{3}(-\dX\d)?)_(?P<adsorbat>\w+)g', fil)
        FE = float(re.findall(r'Free\senergy:\s+(?P<FE>-?\d+\.?\d+)', work_file_dat)[-1])
        unit_cell = re.findall(
            r'(?:1.\saxis:\s+\w+\s+)(?P<x1>\d+\.\d+)(?:.*)(?:\s+2.\saxis:\s+\w+\s+\d+\.\d+\s+)(?P<y2>\d+\.\d+)',
            work_file_dat)[-1]
        Cell_surface_area = round(float(unit_cell[0]), 5) * round(float(unit_cell[1]), 5)
        return adsorbate_result(file_name=fil, Free_energy=FE, Cell_surface_area=Cell_surface_area, adsorb_type=adsorbat_match.group('adsorbat'))
    except: return None


class result:
    def __init__(self,fil: str, FE: float, cell_size: float, Cu_nr: float, slab: slab_result, adsorb: dict[str:adsorbate_result]):
        name_format_grouping = re.search(r'(?P<facet>\d{3}(-\dX\d)?)(?:(_(?P<adsorbate>\w+)_(?P<Adsorbate_nr>\d+)per(?P<size>\d+))+\w?)', fil)
        adsorbat_match = tuple(re.finditer(r'_(?P<adsorbate>[a-zA-Z]+)_(?P<adsorbate_nr>\d+)per(?P<size>\d+)', fil))
        self.file_name = fil
        self.Free_energy = FE
        self.Cell_surface_area = cell_size
        self.facet = name_format_grouping.group('facet')
        self.adsorb_type: str = '|'.join((ad_match.group('adsorbate') for ad_match in adsorbat_match))
        self.Adsorbate_energy: str = '|'.join(ad_match.group('adsorbate')+str(adsorb[ad_match.group('adsorbate')].Free_energy) for ad_match in adsorbat_match)
        self.CO_nr = {ad_match.group('adsorbate'): int(ad_match.group('adsorbate_nr')) for ad_match in adsorbat_match}
        self.theta = self.CO_nr[adsorbat_match[0].group('adsorbate')] / int(adsorbat_match[0].group('size'))
        self.CO_density = self.CO_nr[adsorbat_match[0].group('adsorbate')] / self.Cell_surface_area
        self.N_Cu = Cu_nr
        self.slab_res = slab
        self.adsorb_dict: dict[str:adsorbate_result] = adsorb
        adsorbat_E = sum(2 * self.adsorb_dict[ad_key].Free_energy * self.CO_nr[ad_key] for ad_key in self.adsorb_dict.keys() if ad_key in self.adsorb_type)
        self.binding_E = (self.Free_energy - slab.Free_energy - adsorbat_E) / (2 * sum(self.CO_nr.values()))

def gpaw_out_reader(fil: str, slab_seq: Sequence[slab_result], adsorbate_dic:dict[str:adsorbate_result]) -> result | None:
    try:
        with open(fil,'r') as work_file:
            work_file_dat = work_file.read()
    except FileNotFoundError: return None

    name_format_grouping = re.search(r'(?P<facet>\d{3}(-\dX\d)?)(?:(_(?P<adsorbate>\w+)_(?P<Adsorbate_nr>\d+)per(?P<size>\d+))+\w?)', fil)

    position_str = re.search(r'Positions:\n(\s+\d+\s\w+.*\n)+(?:\nUnit cell:)',work_file_dat)
    Cu_nr = position_str.group().count('Cu')
    nr_at = re.search(r'(?:Number of atoms: )(?P<Nr>\d+)', work_file_dat)
    fe_match = re.findall(r'Free\senergy:\s+(?P<FE>-?\d+\.?\d+)', work_file_dat)[-1]
    unit_cell = re.findall(r'(?:1.\saxis:\s+\w+\s+)(?P<x1>\d+\.\d+)(?:.*)(?:\s+2.\saxis:\s+\w+\s+\d+\.\d+\s+)(?P<y2>\d+\.\d+)', work_file_dat)[-1]
    cell_area = round(float(unit_cell[0]), 5) * round(float(unit_cell[1]), 5)
    for slab in slab_seq:
        if name_format_grouping.group('facet') == slab.facet and cell_area == slab.Cell_surface_area:
            associated_slab = slab
            break
    else: raise Exception(f'Could not associate a slab with the result of {fil}')
    return result(fil, float(fe_match), cell_area, Cu_nr, associated_slab, adsorbate_dic)

def group_to_dict(ite,key):
    iter_sorted = sorted(ite, key=key)
    dict_res = {}
    for key,val in groupby(iter_sorted,key=key): dict_res.update({key:list(val)})
    return dict_res

def write_csv_file(bindings_res: Sequence[result], slab_res: Sequence[slab_result]):
    with open(f'{bindings_res[0].adsorb_type.replace("|","-")}_energies.csv', 'w', newline='') as work_csv:
        csv_writer = csv.writer(work_csv)
        csv_writer.writerow(['file_name', 'Adsorbate', 'facet', 'surface size of unit cel Å^2', 'nr of Cu', 'nr of CO', 'CO density','CO occupency','Slab energy','Adsorbat energy','Total Free_energy', 'Binding Energy'])
        for res in bindings_res: csv_writer.writerow([res.file_name, res.adsorb_type, res.facet, res.Cell_surface_area, res.N_Cu,
                                                     '|'.join(ad_key + str(res.CO_nr[ad_key]) for ad_key in res.CO_nr.keys())
                                                     , res.CO_density, res.theta, res.slab_res.Free_energy, res.Adsorbate_energy, res.Free_energy, res.binding_E])
    with open(f'{bindings_res[0].adsorb_type.replace("|","-")}_minima_energies.csv', 'w', newline='') as work_csv:
        csv_writer = csv.writer(work_csv)
        csv_writer.writerow(['file_name', 'Adsorbate', 'facet', 'surface size of unit cel Å^2', 'nr of Cu', 'nr of CO', 'CO density','CO occupency', 'Slab energy','Adsorbat energy', 'Total Free_energy', 'Binding Energy'])
        bindings_res_facet_group = group_to_dict(bindings_res,key=attrgetter('facet'))
        for key in bindings_res_facet_group.keys():
            facet_CO_nr_group = group_to_dict(bindings_res_facet_group[key], key = lambda x:round(x.CO_density,3))
            min_facet_val = [min(facet_CO_nr_group[key2], key=attrgetter('binding_E')) for key2 in facet_CO_nr_group.keys()]
            for res in min_facet_val: csv_writer.writerow([res.file_name, res.adsorb_type, res.facet, res.Cell_surface_area, res.N_Cu, '|'.join(ad_key + str(res.CO_nr[ad_key]) for ad_key in res.CO_nr.keys()), res.CO_density, res.theta, res.slab_res.Free_energy, res.Adsorbate_energy, res.Free_energy, res.binding_E])
    with open(f'{bindings_res[0].adsorb_type.replace("|","-")}_Slab_energies.csv', 'w', newline='') as work_csv:
        csv_writer = csv.writer(work_csv)
        csv_writer.writerow(['file_name', 'facet', 'surface size of unit cel Å^2', 'Total Free_energy'])
        for sla in slab_res: csv_writer.writerow([sla.file_name,sla.facet,sla.Cell_surface_area,sla.Free_energy])


def plot_bindings_theta(binding_res: Sequence[result]):
    binding_res_groups = group_to_dict(binding_res,key=attrgetter('facet'))
    fig, ax = plt.subplots()#figsize=(12,12))


    for key in binding_res_groups.keys():
        ax.scatter(*zip(*[(bind.theta, bind.binding_E) for bind in binding_res_groups[key]]), marker='.', label=key)
        facet_CO_nr_group = group_to_dict(binding_res_groups[key], key=lambda x:round(x.theta,3))
        min_facet_val = [min(facet_CO_nr_group[key2], key=attrgetter('binding_E')) for key2 in facet_CO_nr_group.keys()]
        ax.plot(*zip(*[(min_f.theta, min_f.binding_E) for min_f in min_facet_val]),label= f'min of {key}')

    ax.legend()
    #ax.set_yticks([bin.binding_E for bin in binding_res], minor=True)
    #ax.grid(axis='y', which='minor', alpha=0.5)
    ax.set_ylabel('DFT Adsorption energy (ev)')
    ax.set_xlabel(r'$\theta$ of adsorbat %s' % binding_res[0].adsorb_type.split('|')[0])

    fig.savefig(f'adsorption_energy_theta_{binding_res[0].adsorb_type.replace("|","-")}-Cu_' + '_'.join(binding_res_groups.keys()))

def plot_bindings_density(binding_res: Sequence[result]):
    binding_res_groups = group_to_dict(binding_res,key=attrgetter('facet'))
    fig, ax = plt.subplots()#figsize=(12,12))


    for key in binding_res_groups.keys():
        ax.scatter(*zip(*[(bind.CO_density, bind.binding_E) for bind in binding_res_groups[key]]), marker='.', label=key)
        facet_CO_nr_group = group_to_dict(binding_res_groups[key], key=lambda x:round(x.theta,3))
        min_facet_val = [min(facet_CO_nr_group[key2], key=attrgetter('binding_E')) for key2 in facet_CO_nr_group.keys()]
        ax.plot(*zip(*[(min_f.CO_density, min_f.binding_E) for min_f in min_facet_val]),label= f'min of {key}')

    ax.legend()
    #ax.set_yticks([bin.binding_E for bin in binding_res], minor=True)
    #ax.grid(axis='y', which='minor', alpha=0.5)
    ax.set_ylabel('DFT Adsorption energy (ev)')
    ax.set_xlabel(r'density of adsorbat %s' % binding_res[0].adsorb_type.split('|')[0])

    fig.savefig(f'adsorption_energy_density_{binding_res[0].adsorb_type.replace("|","-")}-Cu_' + '_'.join(binding_res_groups.keys()))

def silent_print(work,print_also = None):
    print(work)
    if print_also: print(print_also)
    return work

def main(directory: str = '.', adsorbat_str: Sequence[str] = ('CO',), facet: Optional[Sequence[str]] = None, plot=False, csv_write=False):
    if facet:
        if len(facet)>1: facet_pattern='('+'|'.join(facet)+')'
        else: facet_pattern = facet[0]
    else: facet_pattern = r'\d{3}(-\dX\d)?'

    #make the file name pattern here:
    file_string = facet_pattern+''.join([f'_{ad_str}_\d+per\d+' for ad_str in adsorbat_str])+'\w?\Z'

    slab_list = [f for f in os.listdir(directory) if re.match(r'(?:%s)_slab' % facet_pattern, f)]
    slab_results = [set_slap_object(f'{directory}/{r}/slab_k441') if not 'rerun' in os.listdir(f'{directory}/{r}/.') else set_slap_object(f'{directory}/{r}/rerun/slab_k441') for r in slab_list]
    slab_results: Sequence[slab_result] = [res for res in slab_results if res]

    if adsorbat_str == ['Pb']:
        with open(f'{directory}/bulkPb/Pb.txt', 'r') as workfile:
            work_file_dat = workfile.read()
        FE = float(re.findall(r'Free\senergy:\s+(?P<FE>-?\d+\.?\d+)', work_file_dat)[-1])
        adsorbat_match = 'Pb'
        unit_cell = re.findall(
            r'(?:1.\saxis:\s+\w+\s+)(?P<x1>\d+\.\d+)(?:.*)(?:\s+2.\saxis:\s+\w+\s+\d+\.\d+\s+)(?P<y2>\d+\.\d+)',
            work_file_dat)[-1]
        Cell_surface_area = round(float(unit_cell[0]), 5) * round(float(unit_cell[1]), 5)

        adsorbate_result_dict = {'Pb': adsorbate_result(file_name='Pb.txt',Free_energy=FE,Cell_surface_area=Cell_surface_area,adsorb_type=adsorbat_match)}
        print(adsorbate_result_dict['Pb'])
        adsorbate_result_dict['Pb'].Free_energy /= 4

    elif adsorbat_str == ['Cl']:
        adorbat_file_list = [f for f in os.listdir(directory) if
                             any(re.match(r'\d{3}_%s2g' % ad_str, f) for ad_str in adsorbat_str)]
        adsorbate_result_dict = {}
        for ad_str in adsorbat_str:
            for f in adorbat_file_list:
                if 'rerun' in os.listdir(f'{directory}/{f}'): f += '/rerun'
                if ad_res := set_adsorbate_object(f'{directory}/{f}/{ad_str}2g_k111'):
                    adsorbate_result_dict.update({ad_str: ad_res})
                    adorbat_file_list.remove(f.replace('/rerun', ''))
                    break
            else:
                raise Exception(f'Could not find an output file for {ad_str}')
        adsorbate_result_dict['Cl'].Free_energy /= 2 # since it was calculated as Cl2
        adsorbate_result_dict['Cl'].Free_energy += -1.36 # + 0.8 # to move the reference to Cl^-
        adsorbate_result_dict['Cl'].Free_energy += 8.617333262145 * 10**(-5) * 298.15 * math.log(0.1) # + 0.8 # to move the reference to 0.1 M

    elif adsorbat_str == ['F']:
        adorbat_file_list = [f for f in os.listdir(directory) if any(re.match(r'\d{3}_%s2g' % ad_str, f) for ad_str in adsorbat_str)]
        adsorbate_result_dict = {}
        for ad_str in adsorbat_str:
            for f in adorbat_file_list:
                if 'rerun' in os.listdir(f'{directory}/{f}'): f += '/rerun'
                if ad_res := set_adsorbate_object(f'{directory}/{f}/{ad_str}2g_k111'):
                    adsorbate_result_dict.update({ad_str: ad_res})
                    adorbat_file_list.remove(f.replace('/rerun', ''))
                    break
            else:
                raise Exception(f'Could not find an output file for {ad_str}')
            adsorbate_result_dict['F'].Free_energy /= 2  # since it was calculated as F2
            adsorbate_result_dict['F'].Free_energy += -2.87 # to move the reference to F^-
            adsorbate_result_dict['F'].Free_energy += 8.617333262145 * 10 ** (-5) * 298.15 * math.log(0.1)  # + 0.8 # to move the reference to 0.1 M

    else:
        adorbat_file_list = [f for f in os.listdir(directory) if any(re.match(r'\d{3}_%sg' % ad_str, f) for ad_str in adsorbat_str)]
        adsorbate_result_dict = {}
        for ad_str in adsorbat_str:
            for f in adorbat_file_list:
                if 'rerun' in os.listdir(f'{directory}/{f}'): f += '/rerun'
                if ad_res := set_adsorbate_object(f'{directory}/{f}/{ad_str}g_k111'):
                    adsorbate_result_dict.update({ad_str:ad_res})
                    adorbat_file_list.remove(f.replace('/rerun',''))
                    break
            else: raise Exception(f'Could not find an output file for {ad_str}')

    folders = [f for f in os.listdir(directory) if re.match(file_string, f) and 'bad' not in os.listdir(f'{directory}/{f}')]
    result_list: list[result] = [gpaw_out_reader(f'{directory}/{r}/{"".join(adsorbat_str)}ad_k441', slab_results, adsorbate_result_dict) if not 'rerun' in os.listdir(f'{directory}/{r}/') else gpaw_out_reader(f'{directory}/{r}/rerun/{"".join(adsorbat_str)}ad_k441', slab_results, adsorbate_result_dict) for r in folders]
    result_list = [res for res in result_list if res]

    if csv_write: write_csv_file(result_list, slab_results)
    if plot:
        plot_bindings_theta(result_list)
        plot_bindings_density(result_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-dir',help='Directory to the results.', default='.')
    parser.add_argument('--facet','-f', help='Specific facet(s) of interest, if not called then it will take every facet.', nargs='+')
    parser.add_argument('--plot', '-plot', help='Will create a plot over the binding energies for the different structures.', action='store_true', default=False)
    parser.add_argument('--csv','-csv', help='if called will create a csv table of the data.', action='store_true', default=False)
    parser.add_argument('--Adsorbate','-Ad', help='if called look for the specified adsorbate and if not called will assume it is CO.', default=('CO',), nargs='+')
    args = parser.parse_args()

    main(directory=args.dir, adsorbat_str=args.Adsorbate, facet=args.facet, plot=args.plot, csv_write=args.csv)
