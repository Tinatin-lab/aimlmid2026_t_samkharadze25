import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import re
import time
import os

def fetch_data_from_url(url="https://max.ge/aiml_midterm/14753_html/t_samkharadze25_14753.html"):
    print(f"Fetching data from: {url}")
    print("="*70)
    
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    session.headers.update(headers)
    
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = session.get(url, timeout=15, verify=True)
                response.raise_for_status()
                print("✓ Successfully connected to the server")
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                else:
                    raise
        
        soup = BeautifulSoup(response.content, 'html.parser')
        data_points = []
        
        text_content = soup.get_text()
        coord_patterns = re.findall(r'x:\s*([\d\.-]+)[,\s]+y:\s*([\d\.-]+)', text_content, re.IGNORECASE)
        coord_patterns2 = re.findall(r'\(([\d\.-]+),\s*([\d\.-]+)\)', text_content)
        
        for match in coord_patterns + coord_patterns2:
            try:
                x = float(match[0])
                y = float(match[1])
                data_points.append([x, y])
            except:
                pass
        
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                matches = re.findall(r'\[([\d\.,\s-]+)\]', script.string)
                for match in matches:
                    try:
                        values = [float(v.strip()) for v in match.split(',')]
                        if len(values) >= 2:
                            for i in range(0, len(values)-1, 2):
                                data_points.append([values[i], values[i+1]])
                    except:
                        pass
        
        elements_with_data = soup.find_all(attrs={'data-x': True, 'data-y': True})
        for elem in elements_with_data:
            try:
                x = float(elem.get('data-x'))
                y = float(elem.get('data-y'))
                data_points.append([x, y])
            except:
                pass
        
        if data_points:
            seen = set()
            unique_points = []
            for point in data_points:
                point_tuple = tuple(point)
                if point_tuple not in seen:
                    seen.add(point_tuple)
                    unique_points.append(point)
            data_points = unique_points
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 2:
                    try:
                        x = float(cols[0].text.strip())
                        y = float(cols[1].text.strip())
                        data_points.append([x, y])
                    except:
                        pass
        
        data_divs = soup.find_all(['div', 'span'], class_=re.compile(r'data|point'))
        
        if data_points:
            df = pd.DataFrame(data_points, columns=['X', 'Y'])
            print(f"✓ Successfully extracted {len(df)} data points")
            return df
        
        print("\n⚠ Could not automatically extract data points")
        print("Saving HTML content for manual inspection...")
        
        with open('page_content.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print("HTML saved to: page_content.html")
        print("\nThe data might be embedded in JavaScript or require manual extraction.")
        print("Please check the HTML file or provide the data manually.")
        
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        print("\n⚠ Connection failed. Possible reasons:")
        print("  - Server blocking automated requests")
        print("  - Network connectivity issues")
        print("  - SSL/TLS certificate problems")
        print("\n💡 Solution: Use manual data entry option")
        print("  1. Visit the URL in your browser:")
        print(f"     {url}")
        print("  2. Hover over each blue dot to see coordinates")
        print("  3. Choose 'Option 1' when prompted for manual entry")
        return None

def manual_data_entry():
    print("\n" + "="*70)
    print("MANUAL DATA ENTRY")
    print("="*70)
    print("Please hover over each blue dot on the graph and note the coordinates.")
    print("Enter data points (x,y) one per line. Press Enter on empty line when done.")
    print("Format: x,y (e.g., 10.5,20.3)")
    print("-"*70)
    
    data_points = []
    while True:
        user_input = input("Enter point (x,y) or press Enter to finish: ").strip()
        if not user_input:
            break
        try:
            x, y = map(float, user_input.split(','))
            data_points.append([x, y])
            print(f"  ✓ Added point: ({x}, {y})")
        except:
            print("  ✗ Invalid format. Please use: x,y")
    
    if data_points:
        df = pd.DataFrame(data_points, columns=['X', 'Y'])
        print(f"\n✓ Entered {len(df)} data points")
        return df
    return None

def load_sample_data():
    print("\n⚠ Using sample data for demonstration")
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 3 + np.random.normal(0, 2, 50)
    df = pd.DataFrame({'X': x, 'Y': y})
    return df

def calculate_pearson_correlation(df):
    print("\n" + "="*70)
    print("PEARSON'S CORRELATION ANALYSIS")
    print("="*70)
    
    x = df['X'].values
    y = df['Y'].values
    
    correlation_np = np.corrcoef(x, y)[0, 1]
    correlation_scipy, p_value = stats.pearsonr(x, y)
    correlation_pd = df['X'].corr(df['Y'])
    
    n = len(df)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    covariance = np.cov(x, y)[0, 1]
    
    x_diff = x - mean_x
    y_diff = y - mean_y
    numerator = np.sum(x_diff * y_diff)
    denominator = np.sqrt(np.sum(x_diff**2) * np.sum(y_diff**2))
    correlation_manual = numerator / denominator
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Number of data points (n): {n}")
    print(f"  X - Mean: {mean_x:.4f}, Std Dev: {std_x:.4f}")
    print(f"  Y - Mean: {mean_y:.4f}, Std Dev: {std_y:.4f}")
    print(f"  Covariance: {covariance:.4f}")
    
    print(f"\n📈 Pearson's Correlation Coefficient (r):")
    print(f"  Using NumPy:   r = {correlation_np:.6f}")
    print(f"  Using SciPy:   r = {correlation_scipy:.6f}")
    print(f"  Using Pandas:  r = {correlation_pd:.6f}")
    print(f"  Manual Calc:   r = {correlation_manual:.6f}")
    
    print(f"\n📊 Statistical Significance:")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.001:
        significance = "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "Very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "Significant (p < 0.05)"
    else:
        significance = "Not significant (p >= 0.05)"
    print(f"  Significance: {significance}")
    
    r = correlation_scipy
    if abs(r) < 0.3:
        strength = "Weak"
    elif abs(r) < 0.7:
        strength = "Moderate"
    else:
        strength = "Strong"
    
    direction = "positive" if r > 0 else "negative"
    
    print(f"\n💡 Interpretation:")
    print(f"  Correlation strength: {strength} {direction} correlation")
    print(f"  R² (coefficient of determination): {r**2:.6f}")
    print(f"  This means {r**2*100:.2f}% of variance in Y is explained by X")
    
    return correlation_scipy, p_value, r**2

def create_visualization(df, correlation, r_squared, p_value):
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig = plt.figure(figsize=(14, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    x = df['X'].values
    y = df['Y'].values
    
    ax1.scatter(x, y, color='blue', s=50, alpha=0.6, edgecolors='darkblue', linewidth=1.5, label='Data Points')
    
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Best Fit Line (y = {z[0]:.2f}x + {z[1]:.2f})')
    
    ax1.set_xlabel('X Variable', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Variable', fontsize=12, fontweight='bold')
    ax1.set_title(f'Scatter Plot with Regression Line\nPearson r = {correlation:.4f}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    textstr = f'r = {correlation:.4f}\nR² = {r_squared:.4f}\np = {p_value:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax2 = plt.subplot(2, 2, 2)
    predictions = p(x)
    residuals = y - predictions
    ax2.scatter(predictions, residuals, color='green', s=50, alpha=0.6, edgecolors='darkgreen')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(x, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {x.mean():.2f}')
    ax3.set_xlabel('X Variable', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of X', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(y, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax4.axvline(y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean = {y.mean():.2f}')
    ax4.set_xlabel('Y Variable', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Y', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('correlation_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: correlation_visualization.png")
    
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, color='blue', s=60, alpha=0.6, edgecolors='darkblue', linewidth=1.5)
    plt.plot(x_line, p(x_line), "r-", linewidth=2.5, label='Regression Line')
    plt.xlabel('X Variable', fontsize=14, fontweight='bold')
    plt.ylabel('Y Variable', fontsize=14, fontweight='bold')
    plt.title(f'Scatter Plot: Pearson Correlation r = {correlation:.4f}\nR² = {r_squared:.4f}, p < {p_value:.3f}', 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    
    textstr = f'n = {len(df)}\nr = {correlation:.6f}\nR² = {r_squared:.6f}\np-value = {p_value:.6f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('correlation_scatter_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Simple scatter plot saved: correlation_scatter_plot.png")
    
    plt.close('all')

def save_data(df):
    df.to_csv('data_points.csv', index=False)
    print("✓ Data saved: data_points.csv")

def generate_report(df, correlation, p_value, r_squared):
    report = f"""# Task 1: Finding the Correlation - Report

## Data Source
- **URL**: https://max.ge/aiml_midterm/14753_html/t_samkharadze25_14753.html
- **Number of data points**: {len(df)}

## Pearson's Correlation Coefficient

### Formula
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]

### Calculation Steps
1. Calculate means: x̄ = {df['X'].mean():.6f}, ȳ = {df['Y'].mean():.6f}
2. Calculate deviations: (xi - x̄) and (yi - ȳ)
3. Calculate products: Σ[(xi - x̄)(yi - ȳ)] = {np.sum((df['X'] - df['X'].mean()) * (df['Y'] - df['Y'].mean())):.6f}
4. Calculate squared deviations: Σ(xi - x̄)² = {np.sum((df['X'] - df['X'].mean())**2):.6f}, Σ(yi - ȳ)² = {np.sum((df['Y'] - df['Y'].mean())**2):.6f}
5. Compute correlation: r = {correlation:.6f}

### Result
- **Pearson's Correlation Coefficient (r)**: {correlation:.6f}
- **R²**: {r_squared:.6f}
- **P-value**: {p_value:.6f}

## Visualization

![Correlation Scatter Plot](correlation_scatter_plot.png)

The graph shows data points with regression line and correlation statistics.
"""
    
    with open('TASK1_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✓ Report saved: TASK1_REPORT.md")

def main():
    print("\n" + "="*70)
    print("TASK 1: PEARSON'S CORRELATION ANALYSIS")
    print("="*70)
    
    df = fetch_data_from_url()
    
    if df is None:
        print("\nWould you like to:")
        print("1. Enter data manually (recommended if you have access to the graph)")
        print("2. Use sample data for demonstration")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            df = manual_data_entry()
        else:
            df = load_sample_data()
    
    if df is None or len(df) == 0:
        print("\n✗ No data available for analysis")
        return
    
    save_data(df)
    correlation, p_value, r_squared = calculate_pearson_correlation(df)
    create_visualization(df, correlation, r_squared, p_value)
    generate_report(df, correlation, p_value, r_squared)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n✓ Pearson's Correlation Coefficient: r = {correlation:.6f}")
    print(f"✓ All files saved in current directory")
    print("\nGenerated files:")
    print("  - data_points.csv")
    print("  - correlation_scatter_plot.png")
    print("  - correlation_visualization.png")
    print("  - TASK1_REPORT.md")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
