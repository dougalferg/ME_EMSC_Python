import numpy as np

def calcWeightFunction(wn, options):
    """
    Build weights for the reference spectrum, implemented as hyperbolic tangent functions 
    alternating between values 0 and 1. Choose between 3 or 4 inflection points.

    Input:
    wn       - Wavenumbers corresponding to spectra in correction (1D array)  
    options  - Options for correction (dict), containing keys 'Weights_InflectionPoints' and 'Weights_Kappa':
               'Weights_InflectionPoints': turning points in descending order (list of lists of doubles), 
               each pair corresponds to a chemically active area (weighted to one) 
               'Weights_Kappa': listing steepness of function at corresponding inflection point (list of lists of doubles) 
               Setting 'Weights_Kappa' to 0 gives straight edges, with no smoothing. 

    Output:
    weights  - Weights for applying on reference spectrum in ME-EMSC correction (1D array)
    """
    
    infl1 = options['Weights_InflectionPoints'][0][0]
    infl2 = options['Weights_InflectionPoints'][0][1]
    infl3 = options['Weights_InflectionPoints'][1][0]
    infl4 = options['Weights_InflectionPoints'][1][1] if len(options['Weights_InflectionPoints'][1]) > 1 else None
    
    if infl1 < infl3:
        raise ValueError('Error in weight function: Weights_InflectionPoints should be given in decreasing order!')

    Kappa1 = options['Weights_Kappa'][0][0]
    Kappa2 = options['Weights_Kappa'][0][1]
    Kappa3 = options['Weights_Kappa'][1][0]
    Kappa4 = options['Weights_Kappa'][1][1] if len(options['Weights_Kappa'][1]) > 1 else None
    
    lim = 0.0015
    
    # Set up hyperbolic tangent function
    extensionN = 15
    delta_step = 0.094
    x = np.arange(-extensionN, extensionN + delta_step, delta_step)
    num_points = x.size
    
    def hyptan(a, x_range):
        return (np.tanh((a * x_range)) + 1)*0.5

    # Extend wavenumber region
    dwn = wn[1] - wn[0]
    wn_tmp = np.concatenate([(wn[0] - num_points * dwn) + np.arange(num_points) * dwn, wn, 
                             (wn[-1] + dwn) + np.arange(1, num_points + 1) * dwn])
    
    # Find index of inflection points
    i1 = np.argmin(np.abs(wn_tmp - infl1))
    i2 = np.argmin(np.abs(wn_tmp - infl2))
    i3 = np.argmin(np.abs(wn_tmp - infl3))
    
    x1 = x
    x2 = x
    x3 = x

    if infl4 is None:
        # First patch: ones
        p1 = np.ones(np.abs(i3 - num_points) // 2)
        
        # Find index of end of patch 2, check if overlapping with start of patch 4
        i_end_p2 = len(p1) + num_points
        i_start_p4 = i2 - num_points // 2
        if i_end_p2 >= i_start_p4:
            len_between = i2 - i3  # Find number of points between the two inflection points
            new_extension = x[num_points // 2 + len_between // 2]  # Find the new extension of the patch
            x1 = np.arange(-extensionN, new_extension + delta_step, delta_step)  # Make new x for patch 2
            x2 = np.arange(-new_extension, extensionN + delta_step, delta_step)  # Make new x for patch 4
        
        # Second patch: hyperbolic tangent with infl3 and Kappa3
        p2 = -hyptan(Kappa3, x1) + 1
        
        if p2[-1] > lim:
            raise Warning('Weight function: inflection point no. 2 and 3 are too close.')
        
        # Third patch: zeros
        p3 = np.zeros(i_start_p4 - i_end_p2) if i_start_p4 - i_end_p2 >= 0 else []
        
        i_end_p4 = i2 + num_points // 2
        i_start_p6 = i1 - num_points // 2
        if i_end_p4 >= i_start_p6:
            len_between = i1 - i2  # Find number of points between the two inflection points
            new_extension = x[num_points // 2 + len_between // 2]  # Find the new extension of the patch
            x2 = np.arange(x2[0], new_extension + delta_step, delta_step)  # Make new x for patch 1
            x3 = np.arange(-new_extension, extensionN + delta_step, delta_step)  # Make new x for patch 2
        
        # Fourth patch: hyperbolic tangent with infl2 and Kappa2
        p4 = hyptan(Kappa2, x2)
        
        if p4[0] > lim:
            raise Warning('Weight function: inflection point no. 2 and 3 are too close.')
        elif 1 - p4[-1] > lim:
            raise Warning('Weight function: inflection point no. 1 and 2 are too close.')
        
        # Fifth patch: ones
        p5 = np.ones(i_start_p6 - i_end_p4) if i_start_p6 - i_end_p4 >= 0 else []
        
        # Sixth patch: hyperbolic tangent with infl1 and Kappa1
        p6 = -hyptan(Kappa1, x3) + 1
        
        # Seventh patch: zeros
        p7 = np.zeros(len(wn_tmp) - num_points // 2 - i1)
        
        # Weights
        weights = np.concatenate([p1, p2, p3, p4, p5, p6, p7])
        
        diff = len(weights) - len(wn_tmp)
        if diff > 0:
            weights = weights[: -diff]
        elif diff < 0:
            weights = np.concatenate([weights, np.full(-diff, weights[-1])])
    else:
        i4 = np.argmin(np.abs(wn_tmp - infl4))
        x4 = x
        
        # First patch: zeros
        p1 = np.zeros(np.abs((i4 - num_points // 2)))
        
        # Find index of end of patch 2, check if overlapping with start of patch 4
        i_end_p2 = len(p1) + num_points
        i_start_p4 = i3 - num_points // 2
        if i_end_p2 >= i_start_p4:
            len_between = i3 - i4  # Find number of points between the two inflection points
            new_extension = x[num_points // 2 + len_between // 2]  # Find the new extension of the patch
            x1 = np.arange(-extensionN, new_extension + delta_step, delta_step)  # Make new x for patch 1
            x2 = np.arange(-new_extension, extensionN + delta_step, delta_step)  # Make new x for patch 2
        
        # Second patch: hyperbolic tangent with infl4 and Kappa4
        p2 = hyptan(Kappa4, x1)
        
        if 1 - p2[-1] > lim:
            raise Warning('Weight function: inflection point no. 3 and 4 are too close.')
        
        # Third patch: ones
        p3 = np.ones(i_start_p4 - i_end_p2) if i_start_p4 - i_end_p2 >= 0 else []
        
        i_end_p4 = i3 + num_points // 2
        i_start_p6 = i2 - num_points // 2
        if i_end_p4 >= i_start_p6:
            len_between = i2 - i3  # Find number of points between the two inflection points
            new_extension = x[num_points // 2 + len_between // 2]  # Find the new extension of the patch
            x2 = np.arange(x2[0], new_extension + delta_step, delta_step)  # Make new x for patch 1
            x3 = np.arange(-new_extension, extensionN + delta_step, delta_step)  # Make new x for patch 2
        
        # Fourth patch: hyperbolic tangent with infl3 and Kappa3
        p4 = -hyptan(Kappa3, x2) + 1
        
        if 1 - p4[0] > lim:
            raise Warning('Weight function: inflection point no. 3 and 4 are too close.')
        elif p4[-1] > lim:
            raise Warning('Weight function: inflection point no. 2 and 3 are too close.')
        
        # Fifth patch: zeros
        p5 = np.zeros(i_start_p6 - i_end_p4) if i_start_p6 - i_end_p4 >= 0 else []
        
        i_end_p6 = i2 + num_points // 2
        i_start_p8 = i1 - num_points // 2
        if i_end_p6 >= i_start_p8:
            len_between = i1 - i2  # Find number of points between the two inflection points
            new_extension = x[num_points // 2 + len_between // 2]  # Find the new extension of the patch
            x3 = np.arange(x3[0], new_extension + delta_step, delta_step)  # Make new x for patch 1
            x4 = np.arange(-new_extension, extensionN + delta_step, delta_step)  # Make new x for patch 2
        
        # Sixth patch: hyperbolic tangent with infl3 and Kappa3
        p6 = hyptan(Kappa2, x3)
        
        if p6[0] > lim:
            raise Warning('Weight function: inflection point no. 2 and 3 are too close.')
        elif 1 - p6[-1] > lim:
            raise Warning('Weight function: inflection point no. 1 and 2 are too close.')
        
        # Seventh patch: ones
        p7 = np.ones(i_start_p8 - i_end_p6) if i_start_p8 - i_end_p6 >= 0 else []
        
        # Eight patch: hyperbolic tangent with infl1 and Kappa1
        p8 = -hyptan(Kappa1, x4) + 1
        
        # Nineth patch: zeros
        p9 = np.zeros(len(wn_tmp) - num_points // 2 - i1)
        
        # Weights
        weights = np.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9])
        
        diff = len(weights) - len(wn_tmp)
        if diff > 0:
            weights = weights[: -diff]
        elif diff < 0:
            weights = np.concatenate([weights, np.full(-diff, weights[-1])])
    
    weights = weights[num_points: -num_points]
    
    return weights

