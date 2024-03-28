
# ~~~ Tom Winckelman wrote this; maintained at https://github.com/ThomasLastName/fit

import torch
from tqdm import tqdm
from time import time as now
from math import prod

try:    # ~~~ these functions help control the color of console output
    from quality_of_life.my_base_utils import support_for_progress_bars, my_warn
except: # ~~~ however, if those functions are not available, then let their definitions be trivial (for compatibility)
    import warnings
    my_warn = warnings.warn
    from contextlib import contextmanager
    @contextmanager
    def support_for_progress_bars():
        yield

#
# ~~~ A customizable sub-routine to be passed to the high level function `fit` defined below
def standard_train_step( model, data, loss_fn, optimizer, device, history, training_metrics, just, sig ):
    #
    # ~~~ Unpack data and move it to the desired device
    X = data[0].to(device)
    y = data[1].to(device)
    #
    # ~~~ Enter training mode and compute prediction error
    model.train()           # ~~~ set the model to training mode (e.g., activate dropout)
    loss = loss_fn(model(X),y)
    #
    # ~~~ Backpropagation
    loss.backward()         # ~~~ compute the gradient of loss
    optimizer.step()        # ~~~ apply the gradient to the model parameters
    #
    # ~~~ Record any user-specified metrics
    vals_to_print = { "loss": f"{loss.item():<{just}.{sig}f}" }  # ~~~ here we'll store the value of any user-specified metrices, as well as adding them to history
    if training_metrics is not None:
        for key in training_metrics:
            value = training_metrics[key]( model=model, data=data, loss_fn=loss_fn, optimizer=optimizer )   # ~~~ pass the `required_kwargs` defined above
            history[key].append(value)
            vals_to_print[key] = f"{value:<{just}.{sig}f}"
    #
    # ~~~ Zero out the gradients, exit training mode, and return all metrics
    optimizer.zero_grad()   # ~~~ reset the gradient to zero so that we "start fresh" next time standard_train_step is called
    model.eval()            # ~~~ set the model to evaluation mode (e.g., deactivate dropout)
    return loss, history, vals_to_print

#
# ~~~ Standard training
def fit(
        model,
        dataloader, # ~~~ should be of class torch.utils.data.Dataloader
        loss_fn,
        optimizer,
        train_step  = standard_train_step,
        test_data   = None,
        epochs      = 5,
        verbose     = 2,
        training_metrics         = None,
        test_metrics             = None,
        epochal_training_metrics = None,
        epochal_test_metrics     = None,
        device  = ("cuda" if torch.cuda.is_available() else "cpu"), pbar_desc=None,
        just    = 6,    # ~~~ try to limit each printed value to this many characters
        sig     = 4,    # ~~~ try to print this many significant digits
        history = None,
        halting_condition = None
    ):
    #
    # ~~~ First safety feature: assert that training_metrics, test_metrics, and epochal_metrics are each a dictionary (if not None)
    for metrics in ( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics ):
        assert isinstance(metrics,dict) or (metrics is None)
    #
    # ~~~ Second safety feature: assert that no two dictionaries use the same key name
    unique_metric_key_names = set()
    repeat_metric_key_names = list()
    for metrics in ( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics ):
        if metrics is not None:
            unique_metric_key_names = unique_metric_key_names | set(metrics.keys())
            repeat_metric_key_names = repeat_metric_key_names + list(metrics.keys())
    if len(repeat_metric_key_names)==len(unique_metric_key_names):
        metric_key_names = repeat_metric_key_names
    else:
        raise ValueError("key conflict: metrics in different dictionaries are referred to by the same key")
    #
    # ~~~ Third safety feature: assert that "loss", "epoch", and "time" are not the keys for any user-specified metric
    if "loss" in metric_key_names or "epoch" in metric_key_names or "time" in metric_key_names:
        raise ValueError("key conflict: the keys 'loss' and 'epoch' and 'time' are reserved and may not be used as the key names for a user-specified metric")
    #
    #
    # ~~~ Fourth safety feature: require test data in order to use testing metrics
    if test_data is None and ( (test_metrics is not None) or (epochal_test_metrics is not None) ):
        raise ValueError("test_metrics requires test_data")
    #
    # ~~~ Create (or contine) the dictionary called `history` to store the value of the loss, along with any user-supplied metrics
    no_history = (history is None)  # ~~~ whether or not we were given some priori history
    if no_history:
        #
        # ~~~ If we were *not* given any prior history, then create a blank dictionary called `hisory`
        history = { "loss":[] , "epoch":[], "time":[] }
        for key in metric_key_names:
            history[key] = []
    else:
        #
        # ~~~ If we *were* given a prior history, check that it is a dictionary
        if not isinstance(history,dict):
            raise ValueError("A supplied `history` argument must be a dictionary")
        #
        # ~~~ Check that the given history has keys called "loss" and "epoch" and "time"
        if not ("loss" in history.keys()) and ("epoch" in history.keys()) and ("time" in history.keys()):
            raise ValueError("keyerror: a supplied `history` dictionary must include keys for 'loss' and 'epoch'")
        #
        # ~~~ Check that the fileds of the given history are lists of compatible lengthts
        lengths = []
        for key in history:
            assert isinstance( history[key], list ), "Each field in `history` should be a list"
            lengths.append( len(history[key]) )
        lenghts_are_good = len(set(lengths))<=2
        lenghts_are_good = lenghts_are_good and max(lengths)==len(history["loss"])      # ~~~ one acceptable length is the number of iterations (for non-epochal metrics)
        if len(set(lengths))==2:
            lenghts_are_good = lenghts_are_good and min(lengths)==max(history["epoch"]) # ~~~ another acceptable length is the number of epochs (for epochal metrics)
        if not lenghts_are_good:
            for key in history:
                print(f"key '{key}' has length {len(history[key])}")
            raise ValueError("There is an inconsistency in the lenghts of the lists in the dictionary `history`. Each should either be the number of iterations (the length of history['loss']) or the number of epochs (the maximum of history['epoch'])")
        #
        # ~~~ Attempt to catch/correct the pratcice of changing metrics between calls to `fit` which is prone to procduce data incosistencies
        for j,metrics in enumerate(( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics )):
            if metrics is not None:
                for key in metrics:
                    metric_name = f"`{metrics[key].__code__.co_name}`" if hasattr( metrics[key], "__code__" ) else f"'{key}'"
                    this_metric_is_only_every_epoch = (j>1)
                    #
                    # ~~~ If there are any metrics for which we have no history, retroactively populate the historical data on those metrics (populate with zeros)
                    if key not in history.keys():
                        my_warn(f"User-supplied history does not contain a key matching the key '{key}' of user-supplied metric {metric_name}. Historical data for this key value will be populated with zeros.")
                        appropriate_length = max(history["epoch"]) if this_metric_is_only_every_epoch else len(history["loss"])
                        history[key] = appropriate_length*[0]
                    #
                    # ~~~ If we have historical data on one of the metrics we aren't going to record during the upcoming round of training, compare the historical and upcoming frequency of records
                    else:
                        future_data_will_be_only_every_epoch = (j>1)
                        historical_data_was_only_every_epoch = ( len(history[key])==max(history["epoch"]) and not len(history[key])==len(history["loss"]) )
                        if not historical_data_was_only_every_epoch==future_data_will_be_only_every_epoch:  # ~~~ if not either both true or both false, i.e., if the frequencies don't match
                            freq_of_historical_data = "epoch" if historical_data_was_only_every_epoch else "iteration"
                            freq_of_future_data     = "epoch" if future_data_will_be_only_every_epoch else "iteration"
                            my_warn(f"IMPORTANT! User-supplied historical data on key '{key}' was avaliable for every {freq_of_historical_data}. However, it is to be collected every {freq_of_future_data} going forward. The change in the frequency at which this metric is measured will result in a data anomaly in the history dictionary. Please handle this after training is complete (e.g., by constant interpolation).")
        #
        # ~~~ If there is historical data on any metric that we will *not* track in the upcoming round of training, then extend the historical data by populating with zeros
        for key in history:
            if (key not in metric_key_names) and (key not in ("loss","epoch","time")):
                n_data_per_epoch_on_this_key = len(history[key])/max(history['epoch'])
                assert n_data_per_epoch_on_this_key==int(n_data_per_epoch_on_this_key), f"Why isn't the length of data for {key} divisible by the number of epochs!?"
                history[key] += int(n_data_per_epoch_on_this_key)*[0]*epochs
                my_warn(f"User-supplied history contains a key '{key}' which does not match the key of any user-supplied metric. The historical data has been extended by populating it with zeros.")
    #
    # ~~~ Validate that all metrics meet some assumptions upon which the present code is positted: namely, that metric keys are unique, and all keys support the `required_kwargs`
    required_kwargs = { "model", "data", "loss_fn", "optimizer" }   # ~~~ below, we'll call `metric( model=model, data=data, loss_fn=loss_fn, optimizer=optimizer )`
    #
    # ~~~ For all user-supplied metrics...
    for j,metrics in enumerate(( training_metrics, test_metrics, epochal_training_metrics, epochal_test_metrics )):
        #
        # ~~~ ... (if any) ...
        if metrics is not None:
            for key in metrics:
                #
                # ~~~ ... check that the function metrics[key] accepts as a keyword argument each of the `required_kwargs` (idk fam chat gpt came up with the next two lines)
                metric_name = f"`{metrics[key].__code__.co_name}`" if hasattr( metrics[key], "__code__" ) else "metric_function"
                try:
                    #
                    # ~~~ Attempt a generic workaround in order to allow metrics not only to be functions but, also, potentially to be any object with a __call__ method
                    metric_name = f"`{metrics[key].__code__.co_name}`" if hasattr( metrics[key], "__code__" ) else "metric_function"
                    if not hasattr( metrics[key], "__code__" ):
                        metrics[key].__code__ = metrics[key].__call__.__code__
                    vars = metrics[key].__code__.co_varnames[:metrics[key].__code__.co_argcount]    # ~~~ a list of text strings: the names of the arguments that metric_name==metrics[key] accepts
                    accepts_arbitrary_kwargs = bool(metrics[key].__code__.co_flags & 0x08)          # ~~~ whether or not metrics[key] was defined with a `**kwargs` catchall
                    #
                    # ~~~ Specifically, for each of the keyword arguments `required_kwargs` that every metric is assumed to accept, ...
                    for this_required_kwarg in required_kwargs:
                        #
                        # ~~~ ... if this metric does not aaccept that keyword argument, ...
                        if not (accepts_arbitrary_kwargs or (this_required_kwarg in vars)):
                            #
                            # ~~~ ... then write and raise an error message complaining about it
                            vars = ','.join(vars)   # ~~~ a single text string: the names of the arguments that his metric accepts which we concatenate and separate by commas
                            requirement = f"All metrics must support the keyword arguments {required_kwargs} (though they needn't be used in the body of the metric)."
                            violation = f"Please modify the definition of {metric_name} to accept the keyword argument {this_required_kwarg}"
                            suggestion = f", e.g., replace `def {metric_name}({vars})` by `def {metric_name}({vars},**kwargs)`" 
                            raise ValueError( requirement+" "+violation+" "+suggestion )
                except:
                    my_warn(f"Unable to read the arguments of metrics. Please, be aware that an error will occur if a metric does not suppport the arguments {str(required_kwargs)}.")
    #
    # ~~~ Train with a progress bar
    with support_for_progress_bars():
        #
        # ~~~ If verbose==1, then just make one long-ass progress bar for all epochs combined
        if verbose>0 and verbose<2:
            pbar = tqdm( desc=pbar_desc, total=epochs*len(dataloader), ascii=' >=' )
        #
        # ~~~ Regardless of verbosity level, cycle through the epochs
        n_epochs_completed_berforehand = 0 if no_history else max(history["epoch"])
        for n in range(epochs):
            #
            # ~~~ If verbose==2, then create a brand new keras-style progress bar at the beginning of each epoch
            if verbose>=2:
                title = f"Epoch {n+1+n_epochs_completed_berforehand}/{epochs+n_epochs_completed_berforehand}"
                if pbar_desc is not None:
                    title += f" ({pbar_desc})"
                pbar = tqdm( desc=title, total=len(dataloader), ascii=' >=' )
            #
            # ~~~ Cycle through all the batches into which the data is split up
            for j,data in enumerate(dataloader):
                this_is_final_iteration_of_this_epoch = (j+1==len(dataloader))
                #
                # ~~~ Do the actual training (FYI: recording of any user-specified training metrics is intended to occor within the body of train_step; their values will be added to history *and* stored by themselves in vals_to_print which has a number of keys equal 1 greater than the number of user-specified training metrics (if None, then this dictionary's only key will be loss), and each key has a scalar value; the intent is to pass pbar.set_postfix(vals_to_print); in other words, vals_to_print already includes any training_metrics, but the loss; any test_metrics still need to be added to it)
                loss, history, vals_to_print = train_step(
                        model,
                        data,
                        loss_fn,
                        optimizer,
                        device,
                        history,
                        {**training_metrics,**epochal_training_metrics} if (this_is_final_iteration_of_this_epoch and (epochal_training_metrics is not None)) else training_metrics,
                        just, sig
                    )
                #
                # ~~~ No matter what, always record this information (FYI: it is assumed that loss.item() was already added to vals_to_print during train_step)
                history["loss"].append(loss.item())
                history["epoch"].append( n+1 + n_epochs_completed_berforehand )
                history["time"].append(now())
                #
                # ~~~ Regardless of verbosity level, still compute and record any test metrics (we may or may not print them, depending on whether or not verbose>=2)
                if test_metrics is not None:
                    for key in test_metrics:
                        value = test_metrics[key]( model=model, data=test_data, loss_fn=loss_fn, optimizer=optimizer )  # ~~~ pass the `required_kwargs` defined above
                        history[key].append(value)
                        vals_to_print[key] = f"{value:<{just}.{sig}f}"
                #
                # ~~~ Halt early if desired
                if (halting_condition is not None) and halting_condition(history):
                    my_warn("Terminating early because the halting condition has been met.")
                    break
                #
                # ~~~ If 0<verbose<2, then print loss and nothing else
                if verbose>0 and verbose<2:
                    pbar.set_postfix( {"loss":f"{loss.item():<{just}.{sig}f}"} )
                #
                # ~~~ If instead verbose>=2, then print all training and test metrics in addition to the loss
                if verbose>=2:
                    pbar.set_postfix( vals_to_print, refresh=False )
                #
                # ~~~ Whether verbose==1 or verbose==2, update the progress bar each iteration
                if verbose>0:
                    pbar.update()
            #
            # ~~~ At the end of the epoch, regardless of verbosity level, compute any user-specified epochal_test_metrics (the epochal_training_metrics were already computed, and added to both `history` and `vals_to_print` during train_step when `this_is_final_iteration_of_this_epoch`)
            if epochal_test_metrics is not None:
                for key in epochal_test_metrics:
                    value = epochal_test_metrics[key]( model=model, data=test_data, loss_fn=loss_fn, optimizer=optimizer )  # ~~~ pass the `required_kwargs` defined above
                    history[key].append(value)
                    vals_to_print[key] = f"{value:<{just}.{sig}f}"
            #
            # ~~~ At the end of the epoch, if verbose>=2, then finalize the message before closing the progress bar
            if verbose>=2:
                #
                # ~~~ Discard any user-specified training_metrics or training_metrics from vals_to_print (the "loss" will remain, as it is not user-specified) 
                for metrics in (training_metrics,test_metrics):
                    if metrics is not None:
                        for key in metrics:
                            _ = vals_to_print.pop(key)
                #
                # ~~~ Print the final loss of this iteration, along with any use-specified epochal train/test metrics
                pbar.set_postfix( vals_to_print, refresh=False )
                #
                # ~~~ Shut off the progress bar
                pbar.close()
    #
    # ~~~ End
    if verbose>0 and verbose<2:
        pbar.close()
    return history

#
# ~~~ A metric for assessing the phenomenon of vanishing gradients
def count_percent_nonzero_grads( model, data, loss_fn, optimizer, tol=1e-8 ):
    #
    # ~~~ Count the number of gradiens, as well as the number of them which are zero (up to some tolerance)
    amount_nonzero = []
    total_amount = []
    for p in model.parameters():
        total_amount.append( prod(list(p.shape)) )
        amount_nonzero.append( torch.sum(p.grad.abs()>tol).item() )
    #
    # ~~~ Return only the percentage that were non-zero
    return sum(amount_nonzero)/sum(total_amount)


#
# ~~~ Metrics for computing the collective ell^p norm of all model parameters
def compute_ell_p_norm_of_params( model, data, loss_fn, optimizer, p ):
    with torch.no_grad():
        norm_to_the_power_of_p = 0.
        for params in model.parameters():
            norm_to_the_power_of_p += torch.sum(params.abs()**p).item() if isinstance(p,(int,float)) else torch.max(params.abs()).item()
    return norm_to_the_power_of_p**(1/p) if isinstance(p,(int,float)) else norm_to_the_power_of_p

#
# ~~~ The cases p = 1, 2, Inf
compute_ell_1_norm_of_params     = lambda model, data, loss_fn, optimizer: compute_ell_p_norm_of_params( model, data, loss_fn, optimizer, p=1 )
compute_ell_2_norm_of_params     = lambda model, data, loss_fn, optimizer: compute_ell_p_norm_of_params( model, data, loss_fn, optimizer, p=2 )
compute_ell_infty_norm_of_params = lambda model, data, loss_fn, optimizer: compute_ell_p_norm_of_params( model, data, loss_fn, optimizer, p="inf" )

