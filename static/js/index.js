$(document).ready(function() {
    // Initialize carousel if present
    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    };
    var carousels = bulmaCarousel.attach('.results-carousel', options);
    bulmaSlider.attach();
});
