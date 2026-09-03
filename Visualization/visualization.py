import os
import numpy as np
import torch
import plotly.graph_objects as go


def torch_to_numpy(tensor):
    """Converts a PyTorch tensor to a NumPy array if necessary."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def prepare_data(df=None, x=None, y=None):
    """Prepares x and y data from either a DataFrame, lists, or tensors."""
    assert x is not None, 'x must be provided'
    assert y is not None, 'y must be provided'

    if df is not None:
        x = df[x]
        y = df[y]    
    elif isinstance(y, list):
        x = torch_to_numpy(x)
        y = [torch_to_numpy(y_) for y_ in y]
    elif isinstance(y, (torch.Tensor, np.ndarray)):
        x = torch_to_numpy(x)
        if len(y.shape) == 1:
            assert len(y) == len(x), 'x and y must have the same length'
            y = [torch_to_numpy(y)]
        elif len(y.shape) == 2:
            assert y.shape[-1] == len(x), 'x and y must have the same length'
            y = [torch_to_numpy(y[counter, ...]).flatten() for counter in range(y.shape[0])]
        else:
            raise ValueError('y must be a 1D or 2D array')
    else:
        raise ValueError('y must be a list, a torch.Tensor or a np.ndarray')
    
    return x, y


def save_image(fig, save_dir, save_name, save_formats=None, width=600, height=400):
    """Saves a Plotly figure to disk in specified formats and as an HTML file."""
    if save_formats is None:
        save_formats = ['png']

    save_dir_images = os.path.join(save_dir, 'images')
    save_dir_html = os.path.join(save_dir, 'html')
    
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_html, exist_ok=True)

    for fmt in save_formats:
        file_path = os.path.join(save_dir_images, f"{save_name}.{fmt}")
        fig.write_image(
            file_path,
            scale=5,
            width=width,
            height=height
        )    
    
    html_path = os.path.join(save_dir_html, f"{save_name}.html")
    fig.write_html(html_path)


def plot(df=None, x=None, y=None, names=None, highlight_points=None,
         title=None, x_label=None, y_label=None,
         line_width=2, line_colors=None, line_dash=None,
         marker_colors=None, marker_sizes=None,
         xlim=None, ylim=None, mode='lines', opacity=0.7,
         width=600, height=400, template='plotly_white',
         legend_orientation='v', plot_bgcolor='white',
         save_dir=None, save_name='figure', save_formats=None, show=True):
    
    # Initialize mutable defaults safely
    if line_colors is None:
        line_colors = ['royalblue', 'firebrick', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black']
    if line_dash is None:
        line_dash = ['solid', 'dash', 'dashdot', 'dot', 'longdash', 'longdashdot', 'longdashdotdot', 'solid', 'dash', 'dashdot', 'dot']
    if marker_colors is None:
        marker_colors = ['royalblue', 'firebrick', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black']
    if marker_sizes is None:
        marker_sizes = [10] * 11
    if save_formats is None:
        save_formats = ['png']

    # Preparing the data
    x, y = prepare_data(df, x, y)  

    # Creating the figure
    fig = go.Figure()

    # Adding traces to the figure
    for i in range(len(y)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y[i],
                name=names[i] if names is not None else None,
                line=dict(
                    dash=line_dash[i % len(line_dash)],
                    color=line_colors[i % len(line_colors)],
                    width=line_width,
                ),
                marker=dict(    
                    color=marker_colors[i % len(marker_colors)],
                    size=marker_sizes[i % len(marker_sizes)],
                ),
            )
        )
        
    if highlight_points is not None:
        for i in range(len(highlight_points)):
            if isinstance(highlight_points[i], list):
                for j in range(len(highlight_points[i])):
                    pt_idx = highlight_points[i][j]
                    assert 0 <= pt_idx < len(x), 'highlight_points must be a valid index within x'
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[x[pt_idx]],
                            y=[y[i][pt_idx]],
                            mode='markers',
                            marker=dict(
                                color=marker_colors[i % len(marker_colors)],
                                size=marker_sizes[i % len(marker_sizes)],
                                symbol='star',
                            )
                        )
                    )
            elif isinstance(highlight_points[i], dict):
                for key, value in highlight_points[i].items():
                    assert 0 <= value < len(x), 'highlight_points must be a valid index within x'
                    
                    trace_name = f"{names[i]} {key}" if (names and names[i] is not None) else key
                    fig.add_trace(
                        go.Scatter(
                            x=[x[value]],
                            y=[y[i][value]],
                            name=trace_name,
                            mode='markers',
                            marker=dict(
                                color=marker_colors[i % len(marker_colors)],
                                size=marker_sizes[i % len(marker_sizes)],
                                symbol='star',
                            )
                        )
                    )

    # Consolidating layout updates
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            range=xlim,
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='Black',
            linewidth=2,
            ticks='outside',
            tickfont=dict(family='Arial', size=12, color='black'),
            tickwidth=2,
            tickcolor='Black',
            ticklen=5,
            tickangle=0,
        ),
        yaxis=dict(
            range=ylim,
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            linecolor='Black',
            linewidth=2,
            ticks='outside',
            tickfont=dict(family='Arial', size=12, color='black'),
            tickwidth=2,
            tickcolor='Black',
            ticklen=5,
            tickangle=0,
        ),
        width=width,
        height=height,
        template=template,
        plot_bgcolor=plot_bgcolor,
        legend_orientation=legend_orientation,
        showlegend=False
    )
    
    fig.update_traces(mode=mode, opacity=opacity)

    # Saving and showing the figure
    if save_dir is not None:
        save_image(fig, save_dir, save_name, save_formats, width, height)
    if show:
        fig.show()


def HeatMap(x, y, z, title=None, x_label=None, y_label=None, z_label=None,
            colorscale='Jet', reversescale=False, zmin=None, zmax=None,
            width=600, height=400, tickvals=None, ticktext=None,
            template='plotly_white', plot_bgcolor='white',
            save_dir=None, save_name='HeatMap', save_formats=None, show=True):
    """Generates and displays a Plotly HeatMap."""
    if save_formats is None:
        save_formats = ['png']

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z, x=x, y=y,
            colorscale=colorscale,
            reversescale=reversescale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=z_label,
                titleside="right",
                ticklen=5,
                showticklabels=True,
                thickness=25,
                len=0.8,
                titlefont=dict(size=14, family='Arial, sans-serif'),
                tickfont=dict(size=12, family='Arial, sans-serif')
            )
        )
    )
    
    # Consolidating layout configuration
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        width=width,
        height=height,
        template=template,
        plot_bgcolor=plot_bgcolor
    )
    
    if save_dir is not None:
        save_image(fig, save_dir, save_name, save_formats, width, height)
    if show:
        fig.show()


def Bar(x, y, names=None, title=None, x_label=None, y_label=None,
        width=800, height=800, template='plotly_white', plot_bgcolor='white',
        save_dir=None, save_name='Bar', save_formats=None, show=True):
    """Generates and displays a Plotly Bar Chart."""
    if save_formats is None:
        save_formats = ['png']

    fig = go.Figure()
    
    if names is None:
        fig.add_trace(go.Bar(x=x, y=y))
    else:
        for i in range(len(y)):
            fig.add_trace(go.Bar(x=x, y=y[i], name=names[i]))
            
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        template=template,
        plot_bgcolor=plot_bgcolor
    )
    
    if save_dir is not None:
        save_image(fig, save_dir, save_name, save_formats, width, height)
    if show:
        fig.show()


def get_y_lim(y, index=0, y_ranges_scale=None, y_ranges_round=None): 
    """Calculates safe y-limits for a plot given data tensors."""
    if y_ranges_scale is None:
        y_ranges_scale = [[1, 1]]
    if y_ranges_round is None:
        y_ranges_round = [[0, 0]]
        
    y_scale = y_ranges_scale[index]
    y_min, y_max = torch.min(y).item(), torch.max(y).item()
    
    y_range = y_max - y_min
    y_lim = [y_min - y_scale[0] * y_range, y_max + y_scale[1] * y_range]
    y_lim = [round(y_lim[0], y_ranges_round[index][0]), round(y_lim[1], y_ranges_round[index][1])]
    
    if y_lim[0] == y_lim[1]:
        y_lim[0] = y_lim[0] - 1
        y_lim[1] = y_lim[1] + 1
        
    return y_lim


def get_data_plotly(plot_data, down_size=100):
    """Extracts and downsamples visualization data arrays."""
    x = plot_data['x'].cpu().numpy()[::down_size]
    y = plot_data['y'].cpu().numpy()  # We don't downsample the y values
    z = plot_data['z'].cpu().numpy()[:, ::down_size]
    
    print('x shape:', x.shape, 'y shape:', y.shape, 'z shape:', z.shape)
    return x, y, z


def plot_spectrum_comparison(
    wavelength, target_spectrum, pred_spectrum,
    title="Spectrum Comparison", save_dir=None,
    save_name="spectrum_comparison", show=False
):
    """Plots the target and predicted spectra for comparison."""
    # Convert inputs to numpy if they are tensors
    wavelength = torch_to_numpy(wavelength).flatten()
    target_spectrum = torch_to_numpy(target_spectrum).flatten()
    pred_spectrum = torch_to_numpy(pred_spectrum).flatten()

    fig = go.Figure()

    # Add Target Spectrum Trace
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=target_spectrum,
        mode='lines',
        name='Target',
        line=dict(color='black', width=2)
    ))

    # Add Predicted Spectrum Trace
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=pred_spectrum,
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=800,
        height=500
    )

    if save_dir:
        save_image(fig, save_dir, save_name, ['png'], 800, 500)

    if show:
        fig.show()

    return fig