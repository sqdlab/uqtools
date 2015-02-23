import inspect

def fix_args(f=None, **__fixed):
    '''
    return a wrapper to a function with the arguments listed in fixed
    removed from its signature. the wrapper function has the same
    argspec than the wrapped function, less the fixed arguments.
    
    Input:
        f (function) - wrapped function. 
        __fixed (dict) - fixed arguments
    Return:
        wrapper function
    Remarks:
        only functions and bound methods are supported. your mileage may vary
        with other callable objects.
        if f is not given, a decorator function is returned. syntax:
            @fix_args(x=0)
            def f(x, y): 
                ...
    '''
    # if f is not given, return a decorator function
    if f is None:
        return lambda f: fix_args(f, **__fixed)
    # get f's argument list, check for unsupported argument names
    argspec = inspect.getargspec(f)
    args_f = argspec.args if argspec.args is not None else []
    if (('__fixed' in args_f) or ('__defaults' in args_f) or
        (argspec.varargs == '__fixed') or (argspec.varargs == '__defaults') or 
        (argspec.keywords == '__fixed') or (argspec.keywords == '__defaults')):
        raise ValueError('f may not have __fixed and __defaults arguments.')
    # build argument lists
    # - for the function returned to the user (args_in)
    # - supplied to the input function f (args_out)
    args_in = []
    args_out = []
    # copy args including their defaults
    if argspec.defaults is not None:
        __defaults = dict(zip(args_f[-len(argspec.defaults):], argspec.defaults))
    else:
        __defaults = {}
    for idx, arg in enumerate(args_f):
        # assume methods passed to fix_args are bound methods, remove self
        if inspect.ismethod(f) and not idx:
            continue
        # replace fixed positional args by their value
        if arg in __fixed:
            args_out.append('__fixed["{0}"]'.format(arg))
            continue
        # append other args
        if arg in __defaults:
            args_in.append('{0}=__defaults["{0}"]'.format(arg))
        else:
            args_in.append(arg)
        args_out.append(arg)
    # copy varargs
    if argspec.varargs is not None:
        args_out.append('*'+argspec.varargs)
        args_in.append('*'+argspec.varargs)
    # insert fixed kwargs that are not positional args
    for arg in __fixed.keys():
        if arg in args_f:
            # already taken care of
            continue
        if argspec.keywords is None:
            # argument not accepted by f
            raise ValueError('invalid argument {0} to f.'.format(arg))
        args_out.append('{0}=__fixed["{0}"]'.format(arg))
    # copy kwargs
    if argspec.keywords is not None:
        args_out.append('**'+argspec.keywords)
        args_in.append('**'+argspec.keywords)
    # compile and return function
    source_dict = {'in': ', '.join(args_in), 
                   'out': ', '.join(args_out)}
    source = 'def fixed_kwargs_f({in}): return f({out})'.format(**source_dict)
    exec source in locals()
    return fixed_kwargs_f
