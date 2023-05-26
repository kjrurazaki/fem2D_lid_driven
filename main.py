# Finite element code for 2D piecewise Linear Galerkin
# Extended Stokes PDE

from solver import gmres_solver
from imposeBC import imposeBC

from model import Model

import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
from display_results import plot_field_3D
import debug_print

def run_2D(model, method):
      # Impose BCs
      (stiffMat, rhs, u_dir_velocity, 
      u_dir_pressure, boundary_nodes_velocity) = (imposeBC(model, method = method))

      # Convert the numpy array to a sparse matrix
      sparse_stiffMat = csc_matrix(stiffMat)

      # x = np.linalg.solve(stiffMat, rhs.reshape(-1,1))
      x, info, e = gmres_solver(sparse_stiffMat, rhs.reshape(-1,1))
      print(f'info:{info}')
      print(f'e:{e}')

      uh = x.reshape(-1, 1)

      # Residual
      residual = np.linalg.norm(rhs - np.matmul(stiffMat, uh))
      
      # Reshape solution for velocity
      num_rows = uh[:2 * model.lines_A].shape[0] // 2
      uh_reshaped = uh[:2 * model.lines_A].reshape(num_rows, 2)

      residual_bc = np.linalg.norm(u_dir_velocity[boundary_nodes_velocity, :2] - 
                                   uh_reshaped[boundary_nodes_velocity, :])

      return uh[:2 * model.lines_A], uh[2 * model.lines_A:], residual, residual_bc, rhs

if __name__ == '__main__':
      residual_values = []
      residuals_bc_values = []
      i_values = []
      type_values = []
      delta_value = []
      i_list = ['3'] # ['0', '1', '2', '3', '4']
      type_list = ['linear_gls', 'bubble', 'p1-iso-p2'] # ['linear', 'linear_gls', 'bubble', 'p1-iso-p2'] 
      delta_list = [None, 0.001] # [None, 0.001, 0.1, 0.5, 1, 10]
      for i in i_list:
            for type in type_list:
                  for delta in delta_list:
                        if delta is not None and type != 'linear_gls':
                              continue
                        elif delta is None and type == 'linear_gls':
                              continue
                        else:
                              meshdir = f"./Meshes/mesh{i}"
                              method = 'lifting'
                              # Elements type could be linear, bubble or p1-iso-p2
                              model_lid = Model(meshdir, 
                                                delta = delta,
                                                only_dirichlet = True,
                                                element_type = type)
                              
                              uh, p, residual, residual_boundary, rhs = run_2D(model_lid, 
                                                                        method = method)
                              
                              residual_values.append(residual)
                              residuals_bc_values.append(residual_boundary)
                              i_values.append(i)
                              type_values.append(type)
                              delta_value.append(delta)
                              
                              # Print residuals
                              print(f'Residual:{residual}')
                              print(f'Residual BC: {residual_boundary}')

                              model_lid.update_rhs(rhs) # After applying boundary condition - to plot
                              model_lid.update_solution(uh, p)

                              # Printing options for the run
                              dict_print = {
                              'print_f' : 0,
                              'print_rhs' : 0,
                              'print_dirichlet' : 0,
                              'print_neumann' : 0,
                              'print_solution' : 1
                              }
                              #  debug_print.all_prints(model_lid, dict_print)
                             
                              # with open(f'./Models/model_i{i}_type{type}_c{c}_delta{delta}.pkl', 'wb') as f:
                              #       dill.dump(model_stokes, f)
                              
                              # Save results - TODO a more efficient way
                              pd.DataFrame(model_lid.coord).to_csv(f'./Models/coord_i{i}_type{type}_delta{delta}.csv')
                              pd.DataFrame(model_lid.triang_velocity).to_csv(f'./Models/triangv_i{i}_type{type}_delta{delta}.csv')
                              pd.DataFrame(model_lid.triang_pressure).to_csv(f'./Models/triangp_i{i}_type{type}_delta{delta}.csv')
                              pd.DataFrame(model_lid.uh).to_csv(f'./Models/v_i{i}_type{type}_delta{delta}.csv')
                              pd.DataFrame(model_lid.p).to_csv(f'./Models/p_i{i}_type{type}_delta{delta}.csv')
                              pd.DataFrame(model_lid.element.h).to_csv(f'./Models/h_i{i}_type{type}_delta{delta}.csv')
                              
                              # with open(f'./Models/model_i{i}_type{type}_c{c}_delta{delta}.pkl', 'wb') as f:
                              #       dill.dump(model_stokes, f)

      pd.DataFrame.from_dict({'i' : i_values, 
                              'type' : type_values,
                              'delta' : delta_value,
                              'residual' : residual_values,
                              'residual_bc' : residuals_bc_values}).to_csv(f'./Models/residuals.csv')
      # Wait to see graphics
      input("Press [enter] to continue.")