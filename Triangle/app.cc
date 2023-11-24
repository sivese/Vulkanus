#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <cstring>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <optional>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <filesystem>

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	auto isComplete() -> bool { 
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;

	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
	const auto enalbeValidationLayers = false;
#else
	const auto enableValidationLayers = true;
#endif

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

auto CreateDebugUtilsMessengerEXT(
	VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pDebugMessenger
) -> VkResult {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	std::cout << "create debug layer" << std::endl;

	if (func != nullptr)
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

auto DestroyDebugUtilsMessengerEXT(
	VkInstance instance,
	VkDebugUtilsMessengerEXT debugMessenger,
	const VkAllocationCallbacks* pAllocator
) -> void {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	if (func != nullptr)
		func(instance, debugMessenger, pAllocator);
}

class Application {
public:
	auto run() -> void {
		this->init();
		this->mainLoop();
	}

private:
	static std::map<SDL_Window*, Application*> handleTable;

	SDL_Window* window = nullptr;

	VkInstance instance				= VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT debugMessenger;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device					= VK_NULL_HANDLE;
	
	VkQueue graphicsQueue			= VK_NULL_HANDLE;
	VkQueue presentQueue			= VK_NULL_HANDLE;
	
	VkSurfaceKHR surface			= VK_NULL_HANDLE;
	VkSwapchainKHR swapChain		= VK_NULL_HANDLE;

	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;

	VkRenderPass renderPass			= VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline graphicsPipeline		= VK_NULL_HANDLE;

	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool		= VK_NULL_HANDLE;
	//VkCommandBuffer commandBuffer	= VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;

	/* Synchronization objects */
	//VkSemaphore imageAvailableSemaphore;
	//VkSemaphore renderFinishedSemaphore;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;

	std::vector<VkFence> inFlightFences;

	bool framebufferResized = false;
	const int MAX_FRAMES_IN_FLIGHT = 2;
	uint32_t currentFrame;

	//std::string errorMessage;
	std::string appName = "Vulkanus";
	std::string engineName = "V-TWIN";

	bool quit = false;

	/* 
			initialization and create vulkan instance
	*/
	auto init() -> void {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffer();
		createSyncObjects();
	}

	auto createInstance() -> void {
		if (enableValidationLayers && !checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested, but not available!");

		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0)
			throw std::runtime_error(SDL_GetError());

		this->window = SDL_CreateWindow(
			appName.c_str(),				// title
			SDL_WINDOWPOS_CENTERED, // x position
			SDL_WINDOWPOS_CENTERED, // y position
			600, 600,				// width, height value
			SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN	// window flag
		);

		handleTable.insert({ this->window , this });

		SDL_AddEventWatch(framebufferResizeCallback, this->window);

		auto appInfo = VkApplicationInfo {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = appName.c_str(),
			.applicationVersion = VK_MAKE_VERSION(0, 0, 1),
			.pEngineName = engineName.c_str(),
			.engineVersion = VK_MAKE_VERSION(0, 0, 1),
			.apiVersion = VK_API_VERSION_1_3
		};

		auto extensions = getRequiredExtensions();

		auto createInfo = VkInstanceCreateInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext = nullptr,
			.pApplicationInfo = &appInfo,
			.enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
			.ppEnabledExtensionNames = extensions.data()
		};

		auto extensionCount = uint32_t(0);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		auto extensionProperties = std::vector<VkExtensionProperties>(extensionCount);

		std::cout << "available extensions:\n";

		for (const auto& extension : extensionProperties)
			std::cout << '\t' << extension.extensionName << '\n';

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("failed to create instance!");
	}

	auto createSurface() -> void {
		auto result = SDL_Vulkan_CreateSurface(
			this->window, 
			(SDL_vulkanInstance) this->instance,
			(SDL_vulkanSurface*) &(this->surface)
		);

		if (result != SDL_TRUE)
			throw std::runtime_error("failed to create surface");
	}

	/*
		Physical device methods
	*/
	auto pickPhysicalDevice() -> void {
		auto deviceCount = uint32_t(0);
		vkEnumeratePhysicalDevices(this->instance, &deviceCount, nullptr);

		if (deviceCount == 0)
			throw std::runtime_error("failed to find GPUs with Vulkan support!");

		auto devices = std::vector<VkPhysicalDevice>(deviceCount);
		vkEnumeratePhysicalDevices(this->instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				this->physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

		//auto candidates = std::multimap<int, VkPhysicalDevice>();
	}

	auto isDeviceSuitable(VkPhysicalDevice device) -> bool {
		/*
		auto deviceProperties = VkPhysicalDeviceProperties();
		vkGetPhysicalDeviceProperties(device, &deviceProperties);

		auto deviceFeatures = VkPhysicalDeviceFeatures();
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 
			&& deviceFeatures.geometryShader;
		*/
		auto indices = findQueueFamilies(device);
		auto extensionSupported = checkDeviceExtensionSupport(device);

		auto swapChainAdequate = false;

		if (extensionSupported) {
			auto swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionSupported && swapChainAdequate;
	}

	auto checkDeviceExtensionSupport(VkPhysicalDevice device) -> bool {
		auto extensionCount = uint32_t(0);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		auto availableExtensions = std::vector<VkExtensionProperties>(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		auto requiredExtensions = std::set <std::string> (deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions)
			requiredExtensions.erase(extension.extensionName);

		return requiredExtensions.empty();
	}

	auto rateDeviceSuitability(VkPhysicalDevice device) -> int {
		int score = 0;
		
		return score;
	}

	auto findQueueFamilies(VkPhysicalDevice device) -> QueueFamilyIndices {
		auto graphicsFamily = std::optional<uint32_t>();
		std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl;

		graphicsFamily = 0;
		std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl;

		auto indices = QueueFamilyIndices();

		auto queueFamilyCount = uint32_t(0);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		auto queueFamilies = std::vector<VkQueueFamilyProperties>(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		auto i = 0;

		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphicsFamily = i;

			auto presentSupport = VkBool32(false);
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, this->surface, &presentSupport);
			
			if (presentSupport) indices.presentFamily = i;
			
			if (indices.isComplete()) 
				break;

			i++;
		}

		return indices;
	}


	/*
		Logical device methods
	*/
	auto createLogicalDevice() -> void {
		auto indices = findQueueFamilies(this->physicalDevice);

		auto queueCreateInfos = std::vector<VkDeviceQueueCreateInfo>();
		auto uniqueQueueFamilies = std::set<uint32_t>{
			indices.graphicsFamily.value(),
			indices.presentFamily.value()
		};

		auto queuePriority = 1.0f;

		for (auto queueFamily : uniqueQueueFamilies) {
			auto queueCreateInfo = VkDeviceQueueCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamily,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority,
			};

			queueCreateInfos.push_back(queueCreateInfo);
		}

		auto deviceFeatures = VkPhysicalDeviceFeatures();

		auto createInfo = VkDeviceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
			.pQueueCreateInfos = queueCreateInfos.data(),
			.enabledLayerCount = 0,
			.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = &deviceFeatures,
		};

		if (vkCreateDevice(this->physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
			throw std::runtime_error("Failed to create logical device!");

		vkGetDeviceQueue(this->device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(this->device, indices.presentFamily.value(), 0, &presentQueue);
	}


	/*
		Swap Chain
	*/
	auto createSwapChain() -> void {
		auto swapChainSupport = querySwapChainSupport(physicalDevice);

		auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		auto extent = chooseSwapExtent(swapChainSupport.capabilities);

		auto imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
			imageCount = swapChainSupport.capabilities.maxImageCount;

		auto indices = findQueueFamilies(this->physicalDevice);

		uint32_t queueFamilyIndices[] = {
			indices.graphicsFamily.value(),
			indices.presentFamily.value()
		};

		auto createInfo = VkSwapchainCreateInfoKHR {
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = this->surface,
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.preTransform = swapChainSupport.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
			.oldSwapchain = VK_NULL_HANDLE
		};

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		if (vkCreateSwapchainKHR(this->device, &createInfo, nullptr, &(this->swapChain)) != VK_SUCCESS)
			throw std::runtime_error("failed to create swap chain!");

		vkGetSwapchainImagesKHR(this->device, this->swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(this->device, this->swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	auto querySwapChainSupport(VkPhysicalDevice device) -> SwapChainSupportDetails {
		auto details = SwapChainSupportDetails();
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, this->surface, &(details.capabilities));

		auto formatCount = uint32_t(0);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, this->surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, this->surface, &formatCount, details.formats.data());
		}

		auto presentModeCount = uint32_t(0);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, this->surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, this->surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	auto chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) -> VkSurfaceFormatKHR {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return availableFormat;
		}

		return availableFormats.front();
	}

	auto chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) -> VkPresentModeKHR {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				return availablePresentMode;
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	auto chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) -> VkExtent2D {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}

		auto width = 0, height = 0;
		SDL_GetWindowSize(this->window, &width, &height);

		auto actualExtent = VkExtent2D{
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(
			actualExtent.width, 
			capabilities.minImageExtent.width, 
			capabilities.maxImageExtent.width
		);

		actualExtent.height = std::clamp(
			actualExtent.height,
			capabilities.minImageExtent.height,
			capabilities.maxImageExtent.height
		);

		return actualExtent;
	}


	auto createImageViews() -> void {
		swapChainImageViews.resize(swapChainImages.size());

		for (auto i = 0; i < swapChainImages.size(); i++) {
			auto createInfo = VkImageViewCreateInfo{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = swapChainImages[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = swapChainImageFormat,
				.components = VkComponentMapping {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create image views!");
		}

	}

	/*
		!The core rendering parts!
	*/

	auto createRenderPass() -> void {
		auto colorAttachment = VkAttachmentDescription{
			.format = swapChainImageFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp =  VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};

		auto colorAttachmentRef = VkAttachmentReference{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		auto subpass = VkSubpassDescription{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentRef
		};

		auto dependency = VkSubpassDependency{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		};

		auto renderPassInfo = VkRenderPassCreateInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &colorAttachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency
		};

		if (vkCreateRenderPass(this->device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
			throw std::runtime_error("failed to create render pass!");
	}

	auto createGraphicsPipeline() -> void {
		auto vertShaderCode = readFile("vert.spv");
		auto fragShaderCode = readFile("frag.spv");

		auto vertexShaderModule = createShaderModule(vertShaderCode);
		auto fragmentShaderModule = createShaderModule(fragShaderCode);

		auto vertexShaderStageInfo = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertexShaderModule,
			.pName = "main",
		};

		auto fragmentShaderStageInfo = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragmentShaderModule,
			.pName = "main",
		};

		VkPipelineShaderStageCreateInfo shaderStages[] = {
			vertexShaderStageInfo,
			fragmentShaderStageInfo
		};

		auto vertexInputInfo = VkPipelineVertexInputStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 0,
			.pVertexBindingDescriptions = nullptr, //optional field
			.vertexAttributeDescriptionCount = 0,
			.pVertexAttributeDescriptions = nullptr,
		};

		auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		auto viewport = VkViewport {
			.x = 0.0f,
			.y = 0.0f,
			.width = (float) swapChainExtent.width,
			.height = (float) swapChainExtent.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		auto scissor = VkRect2D{
			.offset = {0, 0},
			.extent = swapChainExtent,
		};

		auto dynamicStates = std::vector<VkDynamicState>{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		auto dynamicState = VkPipelineDynamicStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
			.pDynamicStates = dynamicStates.data()
		};

		auto viewportState = VkPipelineViewportStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor
		};

		auto rasterizer = VkPipelineRasterizationStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			.depthBiasConstantFactor = 0.0f,
			.depthBiasClamp = 0.0f,
			.depthBiasSlopeFactor = 0.0f,
			.lineWidth = 1.0f,
		};

		auto multisampling = VkPipelineMultisampleStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
			.minSampleShading = 1.0f,
			.pSampleMask = nullptr,
			.alphaToCoverageEnable = VK_FALSE,
			.alphaToOneEnable = VK_FALSE,
		};

		auto colorBlendAttachment = VkPipelineColorBlendAttachmentState{
			.blendEnable = VK_FALSE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		auto colorBlending = VkPipelineColorBlendStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.logicOp = VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
			.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
		};

		auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 0,
			.pSetLayouts = nullptr,
			.pushConstantRangeCount = 0,
			.pPushConstantRanges = nullptr,
		};

		if (vkCreatePipelineLayout(this->device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");

		auto pipelineInfo = VkGraphicsPipelineCreateInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = nullptr, //optional field
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = this->pipelineLayout,
			.renderPass = this->renderPass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE,
			.basePipelineIndex = -1,
		};
		
		if (vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &(this->graphicsPipeline)) != VK_SUCCESS)
			throw std::runtime_error("failed to create graphics pipeline!");

		vkDestroyShaderModule(this->device, vertexShaderModule, nullptr);
		vkDestroyShaderModule(this->device, fragmentShaderModule, nullptr);
	}

	auto createShaderModule(const std::vector<char>& code) -> VkShaderModule {
		auto createInfo = VkShaderModuleCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.size(),
			.pCode = reinterpret_cast<const uint32_t*>(code.data())
		};

		auto shaderModule = VkShaderModule();

		if (vkCreateShaderModule(this->device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("failed to create shader module!");

		return shaderModule;
	}


	/* Framebuffers */
	auto createFramebuffers() -> void {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (auto i = std::size_t(0); i < swapChainImageViews.size(); i++) {
			VkImageView attachment[] = {
				swapChainImageViews[i]
			};

			auto framebufferInfo = VkFramebufferCreateInfo {
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = this->renderPass,
				.attachmentCount = 1,
				.pAttachments = attachment,
				.width = swapChainExtent.width,
				.height = swapChainExtent.height,
				.layers = 1
			};

			if (vkCreateFramebuffer(this->device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create framebuffer!");
		}
	}

	auto cleanupSwapChain() -> void {
		for (std::size_t i = 0; i < swapChainFramebuffers.size(); i++)
			vkDestroyFramebuffer(this->device, swapChainFramebuffers[i], nullptr);

		for (std::size_t i = 0; i < swapChainImageViews.size(); i++)
			vkDestroyImageView(this->device, swapChainImageViews[i], nullptr);

		vkDestroySwapchainKHR(this->device, this->swapChain, nullptr);
	}

	auto recreateSwapChain() -> void {
		auto width = 0, height = 0;

		SDL_GetWindowSize(this->window, &width, &height);

		auto ev = SDL_Event();

		while (width == 0 || height == 0) {
			SDL_GetWindowSize(this->window, &width, &height);
			SDL_WaitEvent(&ev);
		}

		vkDeviceWaitIdle(this->device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createFramebuffers();
	}

	auto createCommandPool() -> void {
		auto queueFamilyIndices = findQueueFamilies(this->physicalDevice);

		auto poolInfo = VkCommandPoolCreateInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
		};

		if (vkCreateCommandPool(this->device, &poolInfo, nullptr, &(this->commandPool)) != VK_SUCCESS)
			throw std::runtime_error("failed to create command pool!");
	}

	auto createCommandBuffer() -> void {
		this->commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		auto allocInfo = VkCommandBufferAllocateInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = this->commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = static_cast<uint32_t>(this->commandBuffers.size()),
		};

		if (vkAllocateCommandBuffers(this->device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffer");
	}

	auto recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) -> void {
		auto beginInfo = VkCommandBufferBeginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = 0,
			.pInheritanceInfo = nullptr
		};

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
			throw std::runtime_error("failed to begin recording command buffer!");

		auto clearColor = VkClearValue{
			{ {0.0f, 0.0f, 0.0f, 1.0f} }
		};

		auto renderPassInfo = VkRenderPassBeginInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = this->renderPass,
			.framebuffer = swapChainFramebuffers[imageIndex],
			.renderArea = VkRect2D {
				.offset = {0, 0},
				.extent = swapChainExtent,
			},
			.clearValueCount = 1,
			.pClearValues = &clearColor,
		};

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			auto viewport = VkViewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = static_cast<float>(swapChainExtent.width),
				.height = static_cast<float>(swapChainExtent.height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			auto scissor = VkRect2D{
				.offset = {0, 0},
				.extent = swapChainExtent,
			};

			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			std::cout << "vulkan command set scissor" << std::endl;

			vkCmdDraw(commandBuffer, 3, 1, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
			throw std::runtime_error("failed to record command buffer!");
	}

	auto createSyncObjects() -> void {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		auto semaphoreInfo = VkSemaphoreCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};

		auto fenceInfo = VkFenceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

		for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(this->device, &semaphoreInfo, nullptr, &(this->imageAvailableSemaphores[i])) != VK_SUCCESS ||
				vkCreateSemaphore(this->device, &semaphoreInfo, nullptr, &(this->renderFinishedSemaphores[i])) != VK_SUCCESS ||
				vkCreateFence(this->device, &fenceInfo, nullptr, &(this->inFlightFences[i])) != VK_SUCCESS)
				throw std::runtime_error("failed to create semaphore!");
		}
	}

	auto drawFrame() -> void {
		std::cout << "draw frame call" << std::endl;

		vkWaitForFences(this->device, 1, &(this->inFlightFences[currentFrame]), VK_TRUE, UINT64_MAX);
		vkResetFences(this->device, 1, &(this->inFlightFences[currentFrame]));

		auto imageIndex = uint32_t(0);
		auto result = vkAcquireNextImageKHR(this->device, this->swapChain, UINT64_MAX, this->imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		vkResetFences(this->device, 1, &inFlightFences[currentFrame]);

		std::cout << "acquired next image, index ->" << imageIndex << std::endl;

		vkResetCommandBuffer(this->commandBuffers[currentFrame], 0);
		
		recordCommandBuffer(this->commandBuffers[currentFrame], imageIndex);

		std::cout << "record command buffer success" << std::endl;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

		auto submitInfo = VkSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = waitSemaphores,
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &(this->commandBuffers[currentFrame]),
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signalSemaphores
		};

		if (vkQueueSubmit(this->graphicsQueue, 1, &submitInfo, this->inFlightFences[currentFrame]) != VK_SUCCESS)
			throw std::runtime_error("failed to submit draw command buffer!");

		VkSwapchainKHR swapChains[] = { this->swapChain };

		auto presentInfo = VkPresentInfoKHR{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signalSemaphores,
			.swapchainCount = 1,
			.pSwapchains = swapChains,
			.pImageIndices = &imageIndex,
			.pResults = nullptr
		};

		result = vkQueuePresentKHR(this->presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed queue present");
		}

		WriteScreenshots();

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	auto mainLoop() -> void {
		auto loop = 0;

		while (quit == false) {
			auto event = SDL_Event();
			std::cout << "main loop, try : " << loop << std::endl;

			while (SDL_PollEvent(&event)) {
				std::cout << "Polling events, " << event.type << std::endl;
				switch (event.type) {
				case SDL_QUIT:
					this->quit = true;
					break;
				}
			}

			drawFrame();
			loop++;
		}

		vkDeviceWaitIdle(this->device);
	}

	/*
			The last step, destroy all thing
	*/
	auto cleanUp() -> void {
		cleanupSwapChain();

		vkDestroyPipeline(this->device, this->graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(this->device, this->pipelineLayout, nullptr);
		
		vkDestroyRenderPass(this->device, this->renderPass, nullptr);

		for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(this->device, this->imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(this->device, this->renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(this->device, this->inFlightFences[i], nullptr);
		}

		for (auto framebuffer : swapChainFramebuffers)
			vkDestroyFramebuffer(this->device, framebuffer, nullptr);
		
		vkDestroyCommandPool(this->device, this->commandPool, nullptr);
		vkDestroyDevice(this->device, nullptr);
		
		vkDestroySurfaceKHR(this->instance, this->surface, nullptr);

		if (enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

		vkDestroyInstance(this->instance, nullptr);

		SDL_DestroyWindow(this->window);
		SDL_Vulkan_UnloadLibrary();
		SDL_Quit();
	}

	int shotTry = 0;

	auto WriteScreenshots() -> void {
		if (shotTry > 3) return;



		shotTry++;
	}

	static auto readFile(const std::string& filename) -> std::vector<char> {
		//for (const auto& entry : std::filesystem::directory_iterator(current))
		//	std::cout << entry.path() << std::endl;

		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("failed to open file!");

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	static auto framebufferResizeCallback(void* data, SDL_Event* ev) -> int {
		if (ev->type == SDL_WINDOWEVENT && ev->window.event == SDL_WINDOWEVENT_RESIZED) {
			auto win = SDL_GetWindowFromID(ev->window.windowID);

			if (win != (SDL_Window*) data) return 0;

			auto app = Application::handleTable[win];

			app->framebufferResized = true;
		}

		return 0;
	}

	auto checkValidationLayerSupport() -> bool {
		auto layerCount = uint32_t(0);
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		auto availableLayers = std::vector<VkLayerProperties>(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (auto layerName : validationLayers) {
			auto layerFound = false;

			for (auto layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	auto getRequiredExtensions() -> std::vector<const char*> {
		auto extensionCount = uint32_t(0);

		SDL_Vulkan_GetInstanceExtensions(this->window, &extensionCount, nullptr);

		const char** extensions = new const char*[extensionCount];

		auto result = SDL_Vulkan_GetInstanceExtensions(this->window, &extensionCount, extensions);

		if (result != SDL_TRUE)
			throw std::runtime_error(SDL_GetError());

		auto ev = std::vector<const char*>(extensions, extensions + extensionCount);

		if (enableValidationLayers)
			ev.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		return ev;
	}

	auto populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) -> void {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData
	) {

		std::cerr << "validation layer : " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
};

std::map<SDL_Window*, Application*> Application::handleTable;

auto main(int argc, char* argv[]) -> int {
	auto app = Application();

	try {
		app.run();
	}
	catch (std::exception exp) {
		std::cout << exp.what() << std::endl;
	}

	return 0;
}