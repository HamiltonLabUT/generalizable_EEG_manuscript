#phn_tools.py
# Get phonetic feature representations from text grid representations

import numpy as np
def convert_phn(stim, new_type="manner"):
	phns= ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 
		   'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 
		   'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 
		   't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
	if new_type=="manner":
		features = { 'r_colored_vowels': ['er','axr'],
				'diphthongs': ['aw','ay','oy'],
				'glides': ['l','el','r','dx','nx'],
				'semivowels': ['w','y','q'],
				'fricatives': ['f','v','th','dh','s','sh','z','zh','hh','hv'],
				'affricates': ['ch','jh'],
				'plosives': ['p','pcl','t','tcl','k','kcl','b','bcl','d','dcl','g','gcl'],
				'nasals': ['m','em','n','en','ng','eng'],
				'silence': ['epi'],
				}
	elif new_type=="place":
		# place
		features = {
				'bilabial': ['b','bcl','p','pcl','m','em'],
				'labiodental': ['f','v'],
				'dental': ['th','dh'],
				'alveolar': ['t','tcl','d','dcl','n','en','r','nx','s','z','el','l','er','ax-r'],
				'postalveolar': ['sh', 'zh','ch'],
				'palatal': ['y'],
				'velar': ['k','kcl','g','gcl','eng'],
				'glottal': ['hh','hv','q'],
				'back_vowels': ['aa','ao','ow','ah','ax','ax-h','uh','ux','uw'],
				'front_vowels': ['iy','ih','ix','ey','eh','ae'],
				'open_vowels': ['aa','ao','ah','ax','ae','aw','ay'],
				'close_vowels': ['uh','ux','uw','iy','ih','ix','ey','eh'],
		}

	elif new_type=="features":
		features = {
				'dorsal': ['sh', 'k', 'kcl', 'g','gcl','eng','ng','ch','jh'],
				'coronal': ['s','z','t','tcl','d','dcl','n','ch','jh'],
				'labial': ['f','v','p','pcl','b','bcl','m','em'],
				'high': ['uh','ux','uw','iy','ih','ix','ey','eh'],
				'front': ['iy','ih','ix','ey','eh','ae'],
				'low': ['aa','ao','ah','ax','ae','aw','ay'],
				'back': ['aa','ao','ow','ah','ax','ax-h','uh','ux','uw'],
				'plosive': ['p','pcl','t','tcl','k','kcl','b','bcl','d','dcl','g','gcl'],
				'fricative': ['f','v','th','dh','s','sh','z','zh','hh','hv','ch','jh'],
				'syllabic': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux'],
				'nasal': ['m','em','n','en','ng','eng','nx'],
				'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx'],
				'obstruent': ['b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx','f', 'g', 'gcl', 'hh', 'hv','jh', 'k', 'kcl', 'p', 'pcl', 'q', 's', 'sh','t', 'tcl', 'th','v','z', 'zh'],
				'sonorant': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','m', 'n', 'ng', 'eng', 'nx','en','em'],
		}

	new_stim = np.zeros((stim.shape[0], len(features.keys())))
	
	if new_type == "features":
		fkeys = ['sonorant','obstruent','voiced','back','front','low','high','dorsal','coronal','labial','syllabic','plosive','fricative','nasal']
	else:
		fkeys = features.keys()

	for t in np.arange(stim.shape[0]): # loop through time
		if np.sum(stim[t,:])>0:
			phn_num = [i for i,x in enumerate(stim[t,:]) if x == 1]
			phoneme = phns[phn_num[0]]
			for ki, k in enumerate(fkeys):
				if phoneme in features[k]:
					new_stim[t,ki]=1
					#print("%s is a %s, index = %d"%(phoneme, k, ki))
	for ki, k in enumerate(fkeys):
		print(k)

	return new_stim, fkeys



