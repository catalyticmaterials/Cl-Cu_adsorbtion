import csv
import os
import argparse
import re
import matplotlib.pyplot as plt
from operator import attrgetter
from itertools import groupby
import math
from typing import List, Dict, Tuple, Sequence, Callable, Any, NoReturn, Optional
import numpy as np
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go

def JK1MOL1_converter(x: float | int) -> float: return (x*(1/(6.02214076 * (10**23)))) * (6.242 * (10**18))
def kJMOL1_converter(x: float | int) -> float: return (x*(1/(6.242 * (10**21)))) * (6.242 * (10**18))

class constants:
    T = 298.15
    k_boltz = 8.617333262145 * 10**(-5) # ev/K
    Avogadros_number = 6.02214076 * 10**(23) # mol^-1
    # https://janaf.nist.gov/tables/C-093.html
    # gas_TS = -0.615 # ev
    gas_ZPE = {'CO':0.14, 'Cl':0.049/2, 'OH': 0.241, 'F':0.068/2} # ev
    # gas_CpDT = 0.09 # ev
#    gas_CpDT = {'CO':JK1MOL1_converter(8.671) * T,'OH':JK1MOL1_converter(9.172) * T,'Cl':JK1MOL1_converter(6.272)*T,'F':JK1MOL1_converter(6.518)*T} # 8.99 * 10**-5 ev/k
    gas_CpDT = {'CO':kJMOL1_converter(8.671),'OH':kJMOL1_converter(9.172),'Cl':kJMOL1_converter(9.181)/2,'F':kJMOL1_converter(6.518)} # 8.99 * 10**-5 ev/k
    # gas_TS = -0.67
#    gas_TS = {'CO': JK1MOL1_converter(197.142) * T,'OH':JK1MOL1_converter(183.708) * T, 'Cl':JK1MOL1_converter(165.189)*T,'F':JK1MOL1_converter(158.750)*T}  # -0.6130180676813006 ev
    gas_TS = {'CO': JK1MOL1_converter(197.142) * T,'OH':JK1MOL1_converter(183.708) * T, 'Cl':JK1MOL1_converter(223.079)/2*T,'F':JK1MOL1_converter(158.750)*T}  # -0.6130180676813006 ev
    bound_ZPE = {'CO':0.192, 'OH':0.364,'Cl':0.03,'F':0.03}
    bound_CpDT = {'CO':0.076, 'OH':0.046,'Cl':0.053,'F':0.053}
    bound_TS = {'CO':0.153, 'OH':-0.079, 'Cl':0.108,'F':0.108} # ev
    P_zero = 0.1# MPa

@dataclass
class facet():
    file_name: str
    adsorbate: str
    facet: str
    Free_energy: float
    cell_surface_area: float
    adsorbate_no: dict[str:int]
    adsorbate_density: float
    binding_E_zero_U: float
    Cu_nr: int
    theta: float
    Slab_Energy: float
    Adsorbat_Energy: dict[str:float]

    def binding_energy(self, potential: float = 0):
        adsorbat_E = sum(2 * (self.Adsorbat_Energy[ad_key] + potential) * self.adsorbate_no[ad_key] for ad_key in self.Adsorbat_Energy.keys() if ad_key in self.adsorbate)
        binding_E = (self.Free_energy - self.Slab_Energy - adsorbat_E) / (2 * sum(self.adsorbate_no.values()))
        return binding_E


def read_csv_facet(fil: str) -> list[facet]:
    def line_reader(line: list[str]):
        kwargs = {
            'file_name': line[0], #
            'adsorbate': line[1], #
            'facet': line[2], #
            'Free_energy': float(line[10]), #
            'cell_surface_area': float(line[3]), #
            'adsorbate_no': {m.group('ad'): int(m.group('ad_nr')) for m in re.finditer(r'(?P<ad>[a-zA-Z]+)(?P<ad_nr>\d+)', line[5])}, #
            'adsorbate_density': float(line[6]), #
            'binding_E_zero_U': float(line[11]), #
            'Cu_nr': int(line[4]), #
            'theta': float(line[7]), #
            'Slab_Energy': float(line[8]), #
            'Adsorbat_Energy': {m.group('ad'):float(m.group('ad_nr')) for m in re.finditer(r'(?P<ad>\w+)(?P<ad_nr>-?\d+(.\d+)?)',line[9])} #
        }
        return kwargs
    with open(fil, 'r') as work_file:
        csv_dat = csv.reader(work_file)
        return [facet(**line_reader(line)) for line in list(csv_dat)[1:]]


@dataclass
class slab():
    file_name: str
    facet: str
    dft_energy: float
    cell_surface_area: float

def read_CSV_slab(fil: str) -> list[slab]:
    def line_reader(line: list[str]):
        kwargs = {
            'file_name': line[0],
            'facet': line[1],
            'dft_energy': float(line[2]),
            'cell_surface_area': float(line[3])
        }
        return kwargs

    with open(fil,'r') as work_file:
        csv_dat = csv.reader(work_file)
        return [slab(**line_reader(line)) for line in list(csv_dat)[1:]]


def diff_FEuler(Y0: float or int, Y1 : float or int, X_step: float or int) -> float: return (Y1 - Y0)/X_step
def Delta_G(DE_dft: float or int, DZP: float or int ,DTS: float or int,CpDT: float or int) -> float: return DE_dft + DZP - DTS + CpDT
# def G_total(E_adsorp,DZP,TS,CpDT,CO_nr): return E_adsorp * CO_nr + DZP + TS + CpDT

def G_int(E_ad_avg:float,N_CO:int,DZ:float,TS:float,CpDT:float) -> float: return N_CO * (E_ad_avg+DZ-TS+CpDT)

def S_config_diff(theta: float) -> float:
    if theta >= 1: theta = 0.99999999
    return - constants.k_boltz * math.log(theta / (1 - theta))
def S_config_int(theta: float) -> float:
    if theta >= 1: theta = 0.99999999
    return S_config_diff(theta) - (constants.k_boltz/theta) * math.log(1 - theta)

def gibbs_list(sorted_facet: List[facet], potential: float = 0) -> Tuple[List[float], List[float]]:
    Dgibbs, int_gibbs = [], []

    # change to 0.1 molar with Kbt*ln(0.1) as a potential

    for i, point in enumerate(sorted_facet):
        if i == 0: Diff_Bin_E = diff_FEuler(point.theta * point.binding_energy(potential), sorted_facet[i + 1].theta * sorted_facet[i + 1].binding_energy(potential), abs(point.theta - sorted_facet[i + 1].theta))
        else:Diff_Bin_E = diff_FEuler(sorted_facet[i - 1].theta * sorted_facet[i - 1].binding_energy(potential), point.theta * point.binding_energy(potential), abs(point.theta - sorted_facet[i - 1].theta))

        delta_ZPE = constants.bound_ZPE[point.adsorbate] - constants.gas_ZPE[point.adsorbate]
        delta_TS_diff = constants.bound_TS[point.adsorbate] + S_config_diff(point.theta) * constants.T - constants.gas_TS[point.adsorbate]
        TS = constants.bound_TS[point.adsorbate] + S_config_int(point.theta) * constants.T - constants.gas_TS[point.adsorbate]
        delta_Cp = constants.bound_CpDT[point.adsorbate] - constants.gas_CpDT[point.adsorbate]

        Dgibbs.append(Delta_G(DE_dft=Diff_Bin_E, DZP=delta_ZPE, DTS=delta_TS_diff, CpDT=delta_Cp))
        int_gibbs.append((G_int(E_ad_avg=point.binding_energy(potential), N_CO=sum(point.adsorbate_no.values()),DZ=delta_ZPE, TS=TS, CpDT=delta_Cp)) / point.cell_surface_area)
    return int_gibbs,Dgibbs

def liniar_root_interpolation(numerical_func:Sequence[Tuple[float,float]],backwards: bool=False) -> float|None:
    if backwards: numerical_func = reversed(numerical_func)
    for i,point in enumerate(numerical_func):
        if i == 0: continue
        if point[1] == 0:
            root = point[0]
            break
        if (numerical_func[i-1][1] < 0) != (point[1] < 0):
            a = (point[1] - numerical_func[i-1][1])/(point[0] - numerical_func[i-1][0])
            b = - a * numerical_func[i-1][0] + numerical_func[i-1][1]
            root = -b / a
            break
    else: return None
    return root


def y_val_liniar_interpolation(numerical_func: Sequence[Tuple[float,float]], x: float|int) -> float|None:
    for i,point in enumerate(numerical_func):
        if i == 0: continue
        if point[0] == x:
            y_val = point[1]
            break
        if (numerical_func[i-1][0] < x) != (point[0] < x):
            a = (point[1] - numerical_func[i-1][1])/(point[0] - numerical_func[i-1][0])
            b = - a * numerical_func[i-1][0] + numerical_func[i-1][1]
            y_val = a*x+b
            break
    else: return None
    return y_val

def group_to_dict(ite: Sequence, key: Callable[[Any],int|float|str]) -> dict[Any:Any]:
    iter_sorted = sorted(ite, key=key)
    dict_res = {}
    for key,val in groupby(iter_sorted,key=key): dict_res.update({key:list(val)})
    return dict_res

def facet_str_tuple(st:str) -> tuple[int, int, int]: return tuple(int(s) for s in st)
def facet_tuple_str(li: Sequence[int]) -> str: return ''.join(str(l) for l in li)

def folder_exist(folder_name: str) -> NoReturn:
    if folder_name not in os.listdir(): os.mkdir(folder_name)

def plot_integral(facet_dict: dict[str: Sequence[facet]], potential_list: Sequence[float], name: Optional[str] = None):
    sce = 0.248
    plot = go.Figure()
    colour_dic = {(1, 0, 0): '#0000FD',
                  (1, 1, 0): '#00FD00',
                  (1, 1, 1): '#FD0000',
                  (3, 1, 0): '#FD7F00', }

    if facet_dict['111'][0].adsorbate == 'Cl':
        facet_break_point = {
            '100': 7,
            '111': 5
        }
    else: facet_break_point = {}

    marker_styles = ['circle','diamond','square','hexagon','circle-dot','diamond-dot','square-dot','hexagon-dot',]

    for facet_key in facet_dict.keys():
        facet_theta_sorted = sorted(facet_dict[facet_key], key=attrgetter('theta'))
        colour = colour_dic[facet_str_tuple(facet_key)]
        for i,potential in enumerate(potential_list):
            ig_list, dg_list = gibbs_list(facet_theta_sorted, potential)

            breaking_point = facet_break_point[facet_key] if facet_key in facet_break_point else -1

            plot.add_trace(go.Scatter(
            x=[point.adsorbate_density for point in facet_theta_sorted[:breaking_point]],
            y=ig_list[:breaking_point],
            mode='lines+markers',
            line=dict(color=colour),
            marker=dict(color=colour, size=10, symbol=marker_styles[i%len(marker_styles)]),
            #color=colour,
            name=f'({facet_key}) at {potential:.3} V vs SHE'
            ))

    plot.update_layout(
        xaxis_title=f'${facet_dict[facet_key][0].adsorbate}/Å^2$',
        yaxis_title=r'$\Delta\textrm{Gibbs integral energy pr area (eV/Å^2)}$'
    )
    #plot.update_xaxes(title_text=r'Cl/Å^2')
    #plot.update_yaxes(title_text=r'$\Delta$ Gibbs integral energy pr area (eV/Å)')

    if name: plot.write_html(name,include_mathjax='cdn')
    else: plot.show()


def plot_integral_plt(facet_dict: dict[str:Sequence[facet]], potential_list: Sequence[float], name: Optional[str] = None, fun: bool = False):
    if fun: plt.xkcd()
    fig, ax = plt.subplots(figsize=(3.4,2.55),layout='tight',)  # figsize=(12,12)
    sce = 0.248
    colour_dic = {(1, 0, 0): '#0000FD',
                  (1, 1, 0): '#00FD00',
                  (1, 1, 1): '#FD0000',
                  (3, 1, 0): '#FD7F00', }

    marker_styles = ['o','D','s','H']

    if facet_dict['111'][0].adsorbate == 'Cl':
        facet_break_point = {
            '100': 7,
            '111': 5
        }
    else: facet_break_point = {}

    for facet_key in facet_dict.keys():
        facet_theta_sorted = sorted(facet_dict[facet_key], key=attrgetter('theta'))
        colour = colour_dic[facet_str_tuple(facet_key)]
        for i,potential in enumerate(potential_list):
            ig_list, dg_list = gibbs_list(facet_theta_sorted, potential)

            breaking_point = facet_break_point[facet_key] if facet_key in facet_break_point else None
            ax.plot(
                [point.adsorbate_density for point in facet_theta_sorted[:breaking_point]],
                ig_list[:breaking_point],
                label=f'({facet_key}) at {potential-sce:.3} V vs SCE',
                marker='.',#marker_styles[i%len(marker_styles)],
                color=colour
            )

    #    ax.set_ylabel('$\Delta$ Gibbs integral \n energy pr area (eV/$Å^2$)')
    ax.set_ylabel('$\Delta$$G_{nCl}$ /A (eV/$Å^2$)')
    #print(ax.ylabel.fontfamily) # didnt work

    ax.axhline(y=-0.01417896, linestyle='--')
    #ax.legend()
    ax.set_xlabel(r'Cl/$Å^2$')

    ax.text(0.05,0.9,'-0.4V vs SCE', fontsize=8, transform=ax.transAxes,bbox=dict(boxstyle='Square',facecolor='white'))
    ax.text(0.05,0.06,'0.5V vs SCE', fontsize=8, transform=ax.transAxes,bbox=dict(boxstyle='Square',facecolor='white'))

    fig.tight_layout(pad=0.06)

    if name: fig.savefig(name,)
    else: plt.show()


def main(facet_csv_path: str, potential_list: Sequence[float],plt: bool = False,fun: bool = False):
    facet_tuple = read_csv_facet(facet_csv_path)
    facet_group: Dict[str,List[facet]] = group_to_dict(facet_tuple, key=attrgetter('facet'))
    #folder_exist(f'{facet_tuple[0].adsorbate}_vari_potential_plots')

    colour_dic = {(1, 0, 0): '#0000FD',
     (1, 1, 0): '#00FD00',
     (1, 1, 1): '#FD0000',
     (3, 1, 0): '#FD7F00',}

    sce = 0.248

    # -0.152 0.748
    #plot_integral_plt(facet_group, [-0.152,0.748], f'{facet_tuple[0].adsorbate}_gibbs_integral_var_U_plt_pop.svg')
    if plt: plot_integral_plt(facet_group,potential_list, f'{facet_tuple[0].adsorbate}_gibbs_integral_var_U_plt.svg',fun=fun)
    else: plot_integral(facet_group,potential_list, f'{facet_tuple[0].adsorbate}_gibbs_integral_var_U.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('facet_csv',help='path to the facet csv')
    parser.add_argument('-U','--potential',nargs='*', type=float)
    parser.add_argument('-plt','--matplotlib',action='store_true')
    parser.add_argument('-fun','--fun',action='store_true', help='fun works with plt')
    args = parser.parse_args()

    main(args.facet_csv,args.potential,args.matplotlib,args.fun)
