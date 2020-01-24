
class Distn(object):
	"""
	User should define:
	- transform() (theta->theta') or maybe just implement some Domains, which would have default transforms?
	- likelihood() (in terms of original params, for user convenience)
	- sample()
	- names of params

	- mean, mode, whatever (method to call for point prediction)

	- init? (no need I think)
	"""
	def __init__(self, params):
		self.params_ = params

	def __getitem__(self, key):
		return self.__class__(self.params_[:,key])

	def __len__(self):
		return self.params_.shape[1]

	@property
	def params(self):
		return self.transform(self.params_)