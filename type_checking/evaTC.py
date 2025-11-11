class EvaTC:
    '''
    static type checker for eva language
    '''
    def _isNumber(self, exp):
        return isinstance(exp, int) or isinstance(exp, float)

    def _isString(self, exp):
        return isinstance(exp, str)

    def tc(self, exp):
        '''type checks an expression

        Args:
            exp: expression to be type checked

        >>> eva = EvaTC()
        >>> eva.tc(42) == 'number'
        True
        >>> eva.tc("hello") == 'string'
        True
        >>> eva.tc([1, 2, 3]) == 'unknown'
        False
        
          '''
        if self._isNumber(exp):
            return 'number'
        elif self._isString(exp):
            return 'string'
        
        




