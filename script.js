$(function() {
    $('#forecastForm').on('submit', function(event) {
        event.preventDefault();  // Prevent default form submission

        var formData = {
            investment_days: $('#investment_days').val(),
            num_stocks: $('#num_stocks').val(),
            dataset: $('#dataset').val()  // Add dataset selection
        };

        $.ajax({
            type: 'POST',
            url: '/forecast',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            dataType: 'json',
            success: function(response) {
                console.log(response);  // Log the response for debugging

                // Update HTML elements with forecasted values
                $('#forecastValues').text(response.forecasted_values.join(', '));
                $('#suggestion').text(response.suggestion);
                $('#investment').text(response.investment);
                $('#profit').text(response.profit);
                $('#currentPrice').text(response.current_price);  // Display current stock price

                // Show the resultSection div
                $('#resultSection').show();
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.error('Error:', errorThrown);  // Log any errors for debugging

                // Display error message if there's an error
                $('#errorMessage').text('Error: ' + errorThrown);
                $('#errorSection').show();
            }
        });
    });
});
