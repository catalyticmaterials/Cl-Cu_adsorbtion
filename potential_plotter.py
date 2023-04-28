import argparse
import os
import math
import re
import csv
from operator import attrgetter, itemgetter
from itertools import groupby
from dataclasses import dataclass
from typing import List, Dict, Tuple, Sequence, Callable, Any, NoReturn, Optional, Iterable
import numpy as np
from ase.build import bulk
import wulffpack
import matplotlib.pyplot as plt
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
    #gas_CpDT = {'CO':JK1MOL1_converter(8.671) * T,'OH':JK1MOL1_converter(9.172) * T,'Cl':JK1MOL1_converter(6.272)*T,'F':JK1MOL1_converter(6.518)*T} # 8.99 * 10**-5 ev/k
    gas_CpDT = {'CO':kJMOL1_converter(8.671),'OH':kJMOL1_converter(9.172),'Cl':kJMOL1_converter(9.181)/2,'F':kJMOL1_converter(6.518)} # 8.99 * 10**-5 ev/k
    # gas_TS = -0.67
    #gas_TS = {'CO': JK1MOL1_converter(197.142) * T,'OH':JK1MOL1_converter(183.708) * T, 'Cl':JK1MOL1_converter(165.189)*T,'F':JK1MOL1_converter(158.750)*T}  # -0.6130180676813006 ev
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

    def __post_init__(self):
        self.surface_tension = surface_tension(self.Slab_Energy,self.Cu_nr,self.cell_surface_area)

    def binding_energy(self, potential: float = 0):
        adsorbat_E = sum(2 * (self.Adsorbat_Energy[ad_key] + potential) * self.adsorbate_no[ad_key] for ad_key in self.Adsorbat_Energy.keys() if ad_key in self.adsorbate)
        binding_E = (self.Free_energy - self.Slab_Energy - adsorbat_E) / (2 * sum(self.adsorbate_no.values()))
        return binding_E


def read_csv_facet(fil: str) -> list[facet]:
    def line_reader(line: list[str]):
        kwargs = {
            'file_name': line[0],
            'adsorbate': line[1],
            'facet': line[2],
            'Free_energy': float(line[10]),
            'cell_surface_area': float(line[3]),
            'adsorbate_no': {m.group('ad'): int(m.group('ad_nr')) for m in re.finditer(r'(?P<ad>[a-zA-Z]+)(?P<ad_nr>\d+)', line[5])},
            'adsorbate_density': float(line[6]),
            'binding_E_zero_U': float(line[11]),
            'Cu_nr': int(line[4]),
            'theta': float(line[7]),
            'Slab_Energy': float(line[8]),
            'Adsorbat_Energy': {m.group('ad'): float(m.group('ad_nr')) for m in re.finditer(r'(?P<ad>\w+)(?P<ad_nr>-?\d+(.\d+)?)',line[9])}
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


class surface():
    def __init__(self,potential, facet_dict: Dict[str,List[facet]], interpolate: bool = True):
        self._potential = potential
        self._facet_data: Dict[str,List[facet]] = facet_dict
        self._interpolate = interpolate

    @property
    def facet_data(self): return self._facet_data
    @facet_data.setter
    def facet_data(self,val: Dict[str,List[facet]]):
        self._surface_energies = None
        self._surface_concentration = None
        self._wulff = None
        self._facet_data: Dict[str,List[facet]] = val

    @property
    def interpolate(self): return self._interpolate
    @interpolate.setter
    def interpolate(self,val):
        self._surface_energies = None
        self._surface_concentration = None
        self._wulff = None
        self._interpolate = val

    @property
    def potential(self): return self._potential
    @potential.setter
    def potential(self,val):
        self._surface_energies = None
        self._surface_concentration = None
        self._wulff = None
        self._potential = val

    @property
    def surface_energies(self):
        if not hasattr(self,'_surface_energies') or self._surface_energies is None:
            self.compute()
            return self._surface_energies
        return self._surface_energies

    @property
    def surface_concentration(self):
        if not hasattr(self,'_surface_concentration') or self._surface_concentration is None:
            self.compute()
            return self._surface_concentration
        return self._surface_concentration

    @property
    def wulff_shape(self):
        if hasattr(self,'_wulff') and self._wulff is not None:
            return self._wulff
        else:
            lattice_constant = 3.701
            prism = bulk('Cu', a=lattice_constant, crystalstructure='fcc')
            self._wulff = wulffpack.SingleCrystal(self.surface_energies, primitive_structure=prism)
            return self._wulff

    def compute(self):
        surface_energies = {}
        surface_concentraion = {}
        for facet_key in self._facet_data.keys():
            facet_con_sorted = sorted(self._facet_data[facet_key], key=attrgetter('adsorbate_density'))
            gibbs_integral_list, gibbs_differential_list = gibbs_list(facet_con_sorted, self._potential)

            if self._interpolate:
                concentration_equilibrium = liniar_root_interpolation(
                    (*zip([point.adsorbate_density for point in facet_con_sorted], gibbs_differential_list),))
                interface_contribution = y_val_liniar_interpolation(
                    (*zip([point.adsorbate_density for point in facet_con_sorted], gibbs_integral_list),),
                    concentration_equilibrium) if concentration_equilibrium is not None else None
            else:
                energies_under_zero = tuple(
                    (fac, gib_i) for fac, gib_i in zip(facet_con_sorted, gibbs_integral_list) if gib_i <= 0)
                minima = min(energies_under_zero, key=itemgetter(1)) if len(energies_under_zero) > 0 else None
                concentration_equilibrium = minima[0].adsorbate_density if minima is not None else None
                interface_contribution = minima[1] if minima is not None else None

            surf_energy = interface_tension(facet_con_sorted[0].surface_tension,
                                            interface_contribution) if interface_contribution is not None else \
            facet_con_sorted[0].surface_tension
            surface_energies.update({facet_str_tuple(facet_key): surf_energy})
            surface_concentraion.update({facet_str_tuple(facet_key): concentration_equilibrium})
        self._surface_energies = surface_energies
        self._surface_concentration = surface_concentraion


def diff_FEuler(Y0: float or int, Y1 : float or int, X_step: float or int) -> float: return (Y1 - Y0)/X_step
def Delta_G(DE_dft: float or int, DZP: float or int ,DTS: float or int,CpDT: float or int) -> float: return DE_dft + DZP - DTS + CpDT
# def G_total(E_adsorp,DZP,TS,CpDT,CO_nr): return E_adsorp * CO_nr + DZP + TS + CpDT

def G_int(E_ad_avg:float,N_ads:int,DZ:float,TS:float,CpDT:float) -> float: return N_ads * (E_ad_avg+DZ-TS+CpDT)

def S_config_diff(theta: float) -> float:
    if theta >= 1: theta = 0.99999999
    return - constants.k_boltz * math.log(theta / (1 - theta))

def S_config_int(theta: float) -> float:
    if theta >= 1: theta = 0.99999999
    return S_config_diff(theta) - (constants.k_boltz/theta) * math.log(1 - theta)

def gibbs_list(sorted_facet: List[facet], potential: float = 0) -> Tuple[List[float], List[float]]:
    Dgibbs, int_gibbs = [], []

    for i, point in enumerate(sorted_facet):
        if i == 0: Diff_Bin_E = diff_FEuler(point.theta * point.binding_energy(potential), sorted_facet[i + 1].theta * sorted_facet[i + 1].binding_energy(potential), abs(point.theta - sorted_facet[i + 1].theta))
        else: Diff_Bin_E = diff_FEuler(sorted_facet[i - 1].theta * sorted_facet[i - 1].binding_energy(potential), point.theta * point.binding_energy(potential), abs(point.theta - sorted_facet[i - 1].theta))

        delta_ZPE = constants.bound_ZPE[point.adsorbate] - constants.gas_ZPE[point.adsorbate]
        delta_TS_diff = constants.bound_TS[point.adsorbate] + S_config_diff(point.theta) * constants.T - constants.gas_TS[point.adsorbate]
        TS = constants.bound_TS[point.adsorbate] + S_config_int(point.theta) * constants.T - constants.gas_TS[point.adsorbate]
        delta_Cp = constants.bound_CpDT[point.adsorbate] - constants.gas_CpDT[point.adsorbate]

        Dgibbs.append(Delta_G(DE_dft=Diff_Bin_E, DZP=delta_ZPE, DTS=delta_TS_diff, CpDT=delta_Cp))
        int_gibbs.append((G_int(E_ad_avg=point.binding_energy(potential), N_ads=sum(point.adsorbate_no.values()), DZ=delta_ZPE, TS=TS, CpDT=delta_Cp)) / point.cell_surface_area)
    return int_gibbs,Dgibbs


def interface_tension(surface_tension: float, absorption_energy: float) -> float:
    return surface_tension + absorption_energy


def surface_tension(slab_energy: float, cu_nr: float, surface_areal: float) -> float:
    bulk_E = -303.450492
    bulk_nr_Cu = 4
    # print(f'slab E:{slab_energy}; nr of CU: {cu_nr}; areal: {surface_areal}')
    return (slab_energy - (bulk_E * (cu_nr / bulk_nr_Cu))) / (2 * surface_areal)


def liniar_root_interpolation(numerical_func: Sequence[Tuple[float,float]], backwards: bool=False) -> float|None:
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

def y_val_liniar_interpolation(numerical_func:Sequence[Tuple[float,float]], x:float|int) -> float|None:
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

#plots

def plotly_plot_potential_vs_(facet_dat, y_label, save_name: Optional = None):
    colour_dic = {(1, 0, 0): '#0000FD',
                  (1, 1, 0): '#00FD00',
                  (1, 1, 1): '#FD0000',
                  (3, 1, 0): '#FD7F00', }
    sce = 0.248
    plot = go.Figure()
    # it is not possible to have linked axes in plotly like in plt
    for facet_key in facet_dat.keys():
        colour = colour_dic[facet_str_tuple(facet_key)]
        x,y = tuple(zip(*facet_dat[facet_key]))
        plot.add_trace(go.Scatter(
            x=x,y=y,
            mode = 'lines+markers',
            line = dict(color=colour),
            marker = dict(color=colour, size=10),
            name = f'({facet_key})'
        ))

    plot.update_layout(
        xaxis_title=f'potential / V vs SHE',
        yaxis_title=y_label
    )

    if save_name: plot.write_html(save_name,include_mathjax='cdn')
    else: plot.show()


def pop_plot_potential_vs_(facet_dat, y_label, save_name, legend: bool = True, x_lim=None, fun: bool = False):
    if fun: plt.xkcd()
    else: no_fun_allowed = True
    colour_dic = {(1, 0, 0): '#0000FD',
                  (1, 1, 0): '#00FD00',
                  (1, 1, 1): '#FD0000',
                  (3, 1, 0): '#FD7F00', }
    sce = 0.248

    fig, ax = plt.subplots(figsize=(3.4, 2.55), layout='tight')

    for facet_key in facet_dat.keys():
        colour = colour_dic[facet_str_tuple(facet_key)]
        ax.plot(*zip(*facet_dat[facet_key]), label=f'({facet_key})', color=colour)

    if x_lim: ax.set_xlim(x_lim)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('potential / V vs SHE')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(y_label)
    if y_label == 'Surface tension eV/$Å^2$': ax.axhline(y=0, color='black', linestyle='--')

    if legend:  ax.legend()
    sce_ax = ax.secondary_xaxis(location='bottom', functions=(lambda x: x - sce, lambda x: x + sce), xlabel='potential / V vs SCE')
    fig.tight_layout(pad=0.06)
    fig.savefig(save_name)
    return fig


def sprint(x):
    print(x)
    return x

def main(facet_csv_path:str, potential_sce: Sequence[float | int] = (-0.9, 0.6, 500),interpolate: bool = True, draw_wulff: bool = False, plotly: bool = False):
    facet_tuple = read_csv_facet(facet_csv_path)
    facet_group: Dict[str, List[facet]] = group_to_dict(facet_tuple, key=attrgetter('facet'))

    colour_dic = {(1, 0, 0): '#0000FD',
                  (1, 1, 0): '#00FD00',
                  (1, 1, 1): '#FD0000',
                  (3, 1, 0): '#FD7F00', }

    sce = 0.248
    potential_linspace = np.linspace(potential_sce[0] + sce, potential_sce[1] + sce, potential_sce[2])
    #potential_linspace = [-0.33+sce,0.13+sce]
    surfaces = tuple(surface(pot,facet_group,interpolate) for pot in potential_linspace)
    plot_data_con: dict[str: list[Iterable[float, float]]] = {fac: [(surf.potential,surf.surface_concentration[facet_str_tuple(fac)]) for surf in surfaces] for fac in facet_group.keys()}
    plot_data_surf_tens: dict[str: list[Iterable[float, float]]] = {fac: [(surf.potential,surf.surface_energies[facet_str_tuple(fac)]) for surf in surfaces] for fac in facet_group.keys()}

    if plotly:
        plotly_plot_potential_vs_(plot_data_con, f'{facet_tuple[0].adsorbate} concentration ({facet_tuple[0].adsorbate}/$Å^2$)')
        plotly_plot_potential_vs_(plot_data_surf_tens, r'Surface tension eV/$Å^2$')
    else:
        ax_limit = pop_plot_potential_vs_(plot_data_con, f'${facet_tuple[0].adsorbate}^*$ coverage (${facet_tuple[0].adsorbate}^*$/$Å^2$)', f'{facet_tuple[0].adsorbate}_potential_vs_con.svg').get_axes()[0].get_xlim()
        pop_plot_potential_vs_(plot_data_surf_tens, r'Surface tension eV/$Å^2$', f'{facet_tuple[0].adsorbate}_potential_vs_surf_ten.svg', legend=False, x_lim=ax_limit)

    if draw_wulff:
        for surf in surfaces:
            surf.wulff_shape.view(
                         legend=False,
                         save_as=f'{list(surf.facet_data.values())[0][0].adsorbate}_p-{surf.potential}_CU_adsorbate.svg',
                         colors=colour_dic)
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('facet_csv',help='path to the facet csv')
    parser.add_argument('-i','--interpolate', action='store_true', help='Denotes if values between calculations should be guessed at.')
    parser.add_argument('-wulff','--draw_wulff',action='store_true', help='Denotes if Wulff shapes should be drawn.')
    parser.add_argument('-plotly','--plotly',action='store_true', help='if called this will plot via plotly for interactive html plots.')
    args = parser.parse_args()

    main(args.facet_csv,interpolate=args.interpolate,draw_wulff=args.draw_wulff, plotly=args.plotly)