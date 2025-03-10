document.addEventListener('DOMContentLoaded', function() {
    // Sidebar Toggle Elements
    const toggleButton = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    const body = document.body;
    
    // Função para alternar o menu
    function toggleSidebar() {
        sidebar.classList.toggle('show');
        body.classList.toggle('sidebar-open');
    }
    
    // Event listener para o botão
    if (toggleButton) {
        toggleButton.addEventListener('click', toggleSidebar);
    }
    
    // Fechar o menu ao clicar em um link (opcional)
    const menuLinks = document.querySelectorAll('.sidebar .nav-link');
    menuLinks.forEach(link => {
        link.addEventListener('click', function() {
            // Em dispositivos móveis, fechar o menu após clicar
            if (window.innerWidth < 992) {
                sidebar.classList.remove('show');
                body.classList.remove('sidebar-open');
            }
        });
    });
    
    // Fechar o menu ao clicar fora (opcional)
    document.addEventListener('click', function(event) {
        const isClickInsideSidebar = sidebar.contains(event.target);
        const isClickOnToggleButton = toggleButton && toggleButton.contains(event.target);
        
        if (!isClickInsideSidebar && !isClickOnToggleButton && sidebar.classList.contains('show') && window.innerWidth < 992) {
            toggleSidebar();
        }
    });

    // Validação de formulários
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', (event) => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
    
    // Formatter para valores monetários
    const currencyFormatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    });
    
    // Formatter para percentuais
    const percentFormatter = new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    
    // Auto-refresh de preços a cada 5 minutos quando estiver na página principal
    if (window.location.pathname === '/') {
        setInterval(() => {
            fetch('/refresh_prices')
                .then(response => {
                    if (response.ok) {
                        location.reload();
                    }
                })
                .catch(error => console.error('Erro ao atualizar preços:', error));
        }, 300000); // 5 minutos
    }
    
    // Tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Format numbers to currency
    document.querySelectorAll('.format-currency').forEach(element => {
        const value = parseFloat(element.textContent);
        if (!isNaN(value)) {
            element.textContent = currencyFormatter.format(value);
        }
    });
    
    // Format numbers to percentage
    document.querySelectorAll('.format-percent').forEach(element => {
        const value = parseFloat(element.textContent) / 100;
        if (!isNaN(value)) {
            element.textContent = percentFormatter.format(value);
        }
    });
    
    // Handle confirmation dialogs
    document.querySelectorAll('[data-confirm]').forEach(element => {
        element.addEventListener('click', function(e) {
            if (!confirm(this.dataset.confirm)) {
                e.preventDefault();
            }
        });
    });
    
    // Handle tabs
    const tabTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    tabTriggerList.forEach(function (tabTriggerEl) {
        tabTriggerEl.addEventListener('click', function (event) {
            event.preventDefault();
            new bootstrap.Tab(this).show();
        });
    });
    
    // Handle modals
    const modalTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="modal"]'));
    modalTriggerList.forEach(function (modalTriggerEl) {
        modalTriggerEl.addEventListener('click', function (event) {
            event.preventDefault();
            const targetModal = document.querySelector(this.getAttribute('data-bs-target'));
            if (targetModal) {
                const modal = new bootstrap.Modal(targetModal);
                modal.show();
            }
        });
    });
    
    // Handle collapsible elements
    const collapseTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="collapse"]'));
    collapseTriggerList.forEach(function (collapseTriggerEl) {
        collapseTriggerEl.addEventListener('click', function (event) {
            event.preventDefault();
            const targetCollapse = document.querySelector(this.getAttribute('data-bs-target'));
            if (targetCollapse) {
                const collapse = new bootstrap.Collapse(targetCollapse);
                if (targetCollapse.classList.contains('show')) {
                    collapse.hide();
                } else {
                    collapse.show();
                }
            }
        });
    });
    
    // Handle form auto-submission on select change
    document.querySelectorAll('select[data-autosubmit]').forEach(select => {
        select.addEventListener('change', function() {
            this.closest('form').submit();
        });
    });
    
    // Inicializar progress bars animadas
    initProgressBars();
});

// Function to create charts
function createChart(canvasId, type, data, options) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    return new Chart(canvas.getContext('2d'), {
        type: type,
        data: data,
        options: options || {}
    });
}

// Function to fetch data from API
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}

// Function to create progress bars
function initProgressBars() {
    document.querySelectorAll('.progress-bar-animate').forEach(progressBar => {
        const targetWidth = progressBar.getAttribute('data-width') || '0';
        progressBar.style.width = '0%';
        
        setTimeout(() => {
            progressBar.style.width = targetWidth;
        }, 100);
    });
}